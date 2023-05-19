import collections
import logging
import os
import random
from functools import lru_cache
from typing import List, Optional, Tuple
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import logging as hf_logging
import numpy as np
import pytorch_lightning as pl

from ..constants import (
    AUTOMM,
    CHOICES_IDS,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    LABEL,
    LM_TARGET,
    LOGITS,
    MASKS,
    TEMPLATE_LOGITS,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
    CURRENT_TEMPLATE,
)
from .utils import DummyLayer, assign_layer_ids, get_column_features

hf_logging.set_verbosity_error()

logger = logging.getLogger(AUTOMM)


@lru_cache(None)
def warn_once(logger, msg: str):
    logger.warning(msg)

        
class TFewModel(pl.LightningModule):
    """
    Implementation of T-Few (https://arxiv.org/pdf/2205.05638.pdf).
    Refer to https://github.com/r-three/t-few
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str = "bigscience/T0_3B",
        num_classes: Optional[int] = 0,
        length_norm: float = 1.0,  # Normalizes length to adjust for length bias in target template
        unlikely_loss: float = 1.0,  # Adds loss term that lowers probability of incorrect outputs
        mc_loss: float = 1.0,  # Adds multiple choice cross entropy loss
        gradient_checkpointing: Optional[bool] = False,
        pretrained: Optional[bool] = True,
        majority_voting: Optional[bool] = False,
        calibrate_templates = False,
    ):
        """
        Load a pretrained T5-based text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint. We support loading T5ForConditionalGeneration checkpoints from
            Huggingface Models list: https://huggingface.co/models.
            We recommend using T0 backbones. For example, you may use
                - 'bigscience/T0_3B'
                - 'bigscience/T0p'
                - 'bigscience/T0pp'
        num_classes
            The number of classes. 1 for a regression task.
        gradient_checkpointing
            Whether to enable gradient checkpointing
        length_norm
             Normalizes length to adjust for length bias in target template
        unlikely_loss
            Adds loss term that lowers probability of incorrect outputs
        mc_loss
            Adds multiple choice cross entropy loss
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")

        self.checkpoint_name = checkpoint_name

        print("Loading Checkpoint: ", self.checkpoint_name)
        self.num_classes = num_classes

        self.config = AutoConfig.from_pretrained(checkpoint_name)

        if pretrained:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name)
        else:
            if "3B" in checkpoint_name:
                self.model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
            else:
                # self.model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
            # self.model = AutoModelForSeq2SeqLM.from_config(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        self.eos_token = self.tokenizer.eos_token
        self.out_features = (
            self.model.config.hidden_size
        )  # required attribute for some features, e.g. data augmentation

        self.gradient_checkpointing = gradient_checkpointing
        if self.gradient_checkpointing:
            self.dummy_layer = DummyLayer()
            self.model.gradient_checkpointing_enable()

        self.prefix = prefix
        self.padding_token = -100 if "BART0" not in self.checkpoint_name else -1
        self.encoder = self.model.encoder if "BART0" not in self.checkpoint_name else self.model.get_encoder()

        self.mc_loss = mc_loss
        self.unlikely_loss = unlikely_loss
        self.length_norm = length_norm

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

        self.majority_voting = majority_voting

        self.layers_to_update = None
        self.current_template = 0

        self.calibrate_templates = calibrate_templates
        self.nll = []
        self.template_weights=None
        
        self.averaged_logits = {}
        self.calibrate_choices = False
        self.choices = []


    @property
    def text_token_ids_key(self):
        return f"{self.prefix}_{TEXT_TOKEN_IDS}"

    @property
    def text_segment_ids_key(self):
        return f"{self.prefix}_{TEXT_SEGMENT_IDS}"

    @property
    def text_valid_length_key(self):
        return f"{self.prefix}_{TEXT_VALID_LENGTH}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def choices_key(self):
        return f"{self.prefix}_{CHOICES_IDS}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"
    
    @property
    def current_template_key(self):
        return f"{self.prefix}_{CURRENT_TEMPLATE}"

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size


    def set_current_template_in_network(self, current_ensemble_iteration = 0):
        for layer in self.layers_to_update:
            if hasattr(layer, "current_ensemble_iteration"):
                setattr(layer, "current_ensemble_iteration", current_ensemble_iteration)


    def compute_template_weighting(self):
        assert self.nll != None, "No Template NLL computed yet, call compute_template_nll"
        summed_nll = []
        for content in self.nll:
            summed_nll.append(sum(content) / len(content))
        
        summed_nll_tensor = torch.FloatTensor(summed_nll)
        summed_nll_tensor_softmax = torch.nn.functional.softmax(summed_nll_tensor)

        self.set_template_weights(summed_nll_tensor_softmax)

        self.calibrate_templates = False       

    def compute_template_nll(self, batch):
        choices_ids = batch[self.choices_key + "_" + str(0)]
        bs, num_choices = choices_ids.size()[:2]

        label = batch[self.label_key]

        for i in range(100):
            current_content_key = self.text_token_ids_key + "_" + str(i)
            current_choices_key = self.choices_key + "_" + str(i)
            if current_content_key not in batch:
                break
            choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or = self.single_pass(batch, current_content_key, current_choices_key)
            cand_loglikely = -F.cross_entropy(target_template_logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none").view(
                bs, num_choices, -1
            )
            cand_loglikely += (lm_target < 0).view(bs, num_choices, -1) * self.padding_token
            cand_loglikely[range(bs), label] = self.padding_token
            unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != self.padding_token).sum()

            nll = -(F.cross_entropy(choices_scores, label) + unlikely_loss)

            if len(self.nll) > i:
                self.nll[i].append(nll.item())
            else:
                self.nll.append([nll.item()])
        
        return choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or 

    
    def compute_choice_statistics(self):
        import copy
        import spacy
        import string
        import numpy
        from itertools import product

        sp = spacy.load('en_core_web_sm')

        STOP_WORDS = sp.Defaults.stop_words
        STOP_WORDS.remove("no")

        PUNCTUATION = string.punctuation
        averaged_logs = torch.sum(torch.stack(list(self.averaged_logits.values())), dim=0) / len(self.averaged_logits)

        top_k = 10
        all_probs = []
        all_ind = []
        for key, value in self.averaged_logits.items():
            probs = []
            ind = []

            value_nw = value - averaged_logs

            sorted_v, indices = torch.sort(value_nw, descending=True)

            for count, val in enumerate(sorted_v[:1000]):
                if self.tokenizer.decode(indices[count]).lower() in STOP_WORDS or self.tokenizer.decode(indices[count]).lower() in PUNCTUATION or self.tokenizer.decode(indices[count]).lower() == " " or self.tokenizer.decode(indices[count]).lower() in ["<pad>", "<s>", "</s>"]:
                    continue
                probs.append(val.item())
                ind.append(indices[count].item())

            all_probs.append(probs[:top_k])
            all_ind.append(ind[:top_k])
        
        
        same_indices = []
        for i, index in enumerate(all_ind):
            for j, index_2 in enumerate(all_ind):
                if j <= i:
                    continue
                for k, ind in enumerate(index):
                    for l, ind2 in enumerate(index_2):
                        if ind == ind2:
                            same_indices.append((i, j, k, l))

        y_ind = [ele for ele in product(range(0, top_k), repeat = self.num_classes)]

        cont_filter = {x: 1 for x in y_ind}
        for cont in y_ind:
            if any([cont[ele[0]] == ele[2] and cont[ele[1]] == ele[3] for ele in same_indices]):
                cont_filt = 0
            else:
                cont_filt = 1
            cont_filter[cont] = cont_filt

        c=numpy.array([numpy.array(xi) for xi in all_probs])

        all_possible_scores = [sum([c[cl][cont[cl]] for cl in range(self.num_classes)]) * cont_filter[cont] for cont in y_ind]

        highest_score_index = y_ind[all_possible_scores.index(max(all_possible_scores))]

        for cl in range(self.num_classes):
            self.choices.append(self.tokenizer.decode(all_ind[cl][highest_score_index[cl]]))
            

    def single_pass(self, batch, current_content_key, current_choices_key):
        text_token_ids = batch[current_content_key]
        choices_ids = batch[current_choices_key]
        bs, num_choices = choices_ids.size()[:2]

        if self.num_classes > 30 and not self.training:
            midpoint = num_choices // 4

            first_half_choice_ids = choices_ids[:, :midpoint, :]
            second_half_choice_ids = choices_ids[:, midpoint:midpoint*2, :]
            third_half_choice_ids = choices_ids[:, midpoint*2:midpoint*3, :]
            fourth_half_choice_ids = choices_ids[:, midpoint*3:, :]            

            all_choice_scores = []
            all_lm = []
            all_target_template = []
            for half_choice_ids in [first_half_choice_ids, second_half_choice_ids, third_half_choice_ids, fourth_half_choice_ids]:

                half_num_choices = half_choice_ids.shape[1]

                flat_choices_ids = half_choice_ids.flatten(0, 1)

                text_valid_length = batch[self.text_valid_length_key]
                text_masks = (text_token_ids != self.tokenizer.pad_token_id).float()

                inputs_embeds = self.encoder.embed_tokens(text_token_ids)

                if self.gradient_checkpointing:
                    inputs_embeds = self.dummy_layer(inputs_embeds)

                # Forward input through the encoder
                encoder_hidden_states_or = self.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)[0]
                encoder_hidden_states = encoder_hidden_states_or.unsqueeze(dim=1).repeat(1, half_num_choices, 1, 1).flatten(0, 1)

                attention_mask = text_masks.unsqueeze(dim=1).repeat(1, half_num_choices, 1).flatten(0, 1)

                decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)

                decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()

                # Forward encoder output and target template as input for decoder
                model_output = self.model(
                    attention_mask=attention_mask,
                    encoder_outputs=[encoder_hidden_states],
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )

                model_output = model_output.logits

                target_template_logits = model_output  # Decoder Logits over the vocabulary for target template sequence

                lm_target = flat_choices_ids - (- self.padding_token) * (flat_choices_ids == self.tokenizer.pad_token_id).long()

                # Calculate entropy of target templates' logits to target template, i.e. how close the target template is to what
                # the model would predict, going from sentence start token (target_template_logits) to sentence end token (
                # lm_target)
                choices_scores = (
                    F.cross_entropy(target_template_logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                    .view(bs, half_num_choices, -1)
                    .sum(dim=-1)
                )
                # Add length normalization to adjust for target templates of different length
                if self.length_norm > 0:
                    choices_scores = choices_scores / torch.pow(
                        (half_choice_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.length_norm
                    )
                
                # Use the entropy score as the class "logit" scoring of T-Few.
                choices_scores = -choices_scores

                all_choice_scores.append(choices_scores)
                all_lm.append(lm_target)
                all_target_template.append(target_template_logits)

            choices_scores = torch.cat(all_choice_scores, dim=-1)
            lm_target = torch.cat(all_lm)
            target_template_logits = torch.cat(all_target_template)

            return [choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or]               
        else:           
            bs = text_token_ids.size(0)
            # TODO(?) Currently does not support mixed-task batching, but can be added by adjusting the label_templates dict.

            bs, num_choices = choices_ids.size()[:2]
            flat_choices_ids = choices_ids.flatten(0, 1)

            text_valid_length = batch[self.text_valid_length_key]
            text_masks = (text_token_ids != self.tokenizer.pad_token_id).float()

            inputs_embeds = self.encoder.embed_tokens(text_token_ids)

            if self.gradient_checkpointing:
                inputs_embeds = self.dummy_layer(inputs_embeds)

            # Forward input through the encoder
            encoder_hidden_states_or = self.encoder(inputs_embeds=inputs_embeds, attention_mask=text_masks)[0]
            encoder_hidden_states = encoder_hidden_states_or.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)

            attention_mask = text_masks.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)

            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)

            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()

            # Forward encoder output and target template as input for decoder
            model_output = self.model(
                attention_mask=attention_mask,
                encoder_outputs=[encoder_hidden_states],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            model_output = model_output.logits

            target_template_logits = model_output  # Decoder Logits over the vocabulary for target template sequence

            lm_target = flat_choices_ids - (- self.padding_token) * (flat_choices_ids == self.tokenizer.pad_token_id).long()

            # Calculate entropy of target templates' logits to target template, i.e. how close the target template is to what
            # the model would predict, going from sentence start token (target_template_logits) to sentence end token (
            # lm_target)
            choices_scores = (
                F.cross_entropy(target_template_logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                .view(bs, num_choices, -1)
                .sum(dim=-1)
            )
            # Add length normalization to adjust for target templates of different length
            if self.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.length_norm
                )
            
            # Use the entropy score as the class "logit" scoring of T-Few.
            choices_scores = -choices_scores
            
            if self.calibrate_choices:
                with torch.no_grad():
                    decoder_input_ids = torch.zeros_like(flat_choices_ids[:bs, :1])
    
                    model_output_2 = self.model(
                        attention_mask=text_masks,
                        decoder_input_ids=decoder_input_ids,
                        encoder_outputs=[encoder_hidden_states_or],
                    )
                    model_output_log = model_output_2.logits
                    labels = batch[self.label_key]

                    for cl in range(self.num_classes):
                        # print(labels)
                        labels_of_class = labels == cl
                        # print(labels_of_class)
                        labels_indices = torch.flatten(labels_of_class.nonzero())#.tolist()
                        # print(labels_indices)
                        logits_of_class = torch.index_select(model_output_log, 0, labels_indices)
                        # print(logits_of_class.size())
                        # print("-------------")
                        if cl in self.averaged_logits:
                            self.averaged_logits[cl] += torch.nn.functional.softmax(torch.sum(logits_of_class,dim=[0,1]))
                        else:
                            self.averaged_logits[cl] =  torch.nn.functional.softmax(torch.sum(logits_of_class, dim=[0,1]))

            return [choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or]


    def set_template_weights(self, weights):
        self.template_weights = weights

    def forward(
        self,
        batch: dict,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        if self.training:
            num_templates = 0
            for i in range(100):
                current_content_key = self.text_token_ids_key + "_" + str(i)
                if current_content_key not in batch:
                    break
                num_templates +=1

            current_template = batch[self.current_template_key].tolist()[0]

            self.set_current_template_in_network(0)

            current_content_key = self.text_token_ids_key + "_" + str(current_template)
            current_choices_key = self.choices_key + "_" + str(current_template)

            choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or = self.single_pass(batch, current_content_key, current_choices_key)

            self.current_template += 1

        else:
            if self.calibrate_templates:
                    with torch.no_grad():
                        choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or  = self.compute_template_nll(batch)

            logits_collection = []
            if self.majority_voting:
                for i in range(100):
                    current_content_key = self.text_token_ids_key + "_" + str(i)
                    current_choices_key = self.choices_key + "_" + str(i)
                    if current_content_key not in batch:
                        break

                    for curr_it in range(1): #TODO: OUTDATED, remove ensemble part.
                        self.set_current_template_in_network(curr_it) # just for setting the current ensemble
                        choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or = self.single_pass(batch, current_content_key, current_choices_key)
                        if self.template_weights != None:
                            logits_collection.append(self.template_weights[i] * choices_scores)
                        else:
                            logits_collection.append(choices_scores)
            
            else:
                i = batch[self.current_template_key].tolist()[0]

                current_content_key = self.text_token_ids_key + "_" + str(i)
                current_choices_key = self.choices_key + "_" + str(i)
                
                for curr_it in range(1):
                    self.set_current_template_in_network(curr_it) # just for setting the current ensemble
                    choices_scores, model_output, target_template_logits, lm_target, text_valid_length, encoder_hidden_states_or = self.single_pass(batch, current_content_key, current_choices_key)
                    logits_collection.append(choices_scores)

            logits_collection = torch.stack(logits_collection, dim=1)
            majority = torch.sum(logits_collection, dim=1)
            choices_scores = majority


        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        #  FIXME(?) Not sure having column features with the decoder vocabulary logits in T-Few makes sense
        column_features, column_feature_masks = get_column_features(
            batch=batch,
            column_name_prefix=self.text_column_prefix,
            features=model_output,
            valid_lengths=text_valid_length,
        )
        ret[COLUMN_FEATURES][FEATURES].update(column_features)
        ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret.update(
            {
                LOGITS: choices_scores,  # needed for default crossentropy loss
                TEMPLATE_LOGITS: target_template_logits,  # needed for unlikelihood loss
                LM_TARGET: lm_target,  # needed for lm loss
                FEATURES: encoder_hidden_states_or[
                    :, 0, :
                ],  # needed to ensure compatibility to encoder-only pipelines
            }
            )

        return {self.prefix: ret}



    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        if "BART0" in self.checkpoint_name:
            pre_encoder_patterns = (
                "embeddings",
                "LayerNorm",
                "wte",
                "wpe",
                "shared.weight",
                "encoder.conv.conv",
                "dummy_layer",
                "layernorm_embedding", #BART0
                "embed_positions", #BART0
            )
        else:
            pre_encoder_patterns = (
                "embeddings",
                "LayerNorm",
                "wte",
                "wpe",
                "shared.weight",
                "encoder.conv.conv",
                "dummy_layer",
            )
        post_encoder_patterns = ("head", "pooler", "ln_f", "final_layer_norm")
        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 1

        for name, id in name_to_id.items():  # no layer should be assigned zero id as zero id is finetuned
            if id == 0:
                name_to_id[name] = 1

        return name_to_id

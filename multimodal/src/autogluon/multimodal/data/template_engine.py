import logging
import os
import json
import copy

import numpy as np
import random
from omegaconf import OmegaConf

import torch

from autogluon.multimodal.data.templates import DatasetTemplates, Template, TemplateCollection

from ..constants import AUTOMM
from datasets import load_dataset

logger = logging.getLogger(AUTOMM)



class TemplateEngine:
    """
    Class to manage the selection and use of templates.
    """

    def apply_and_save_best_choices(self, save_path, manual_choice=None):
        # assert len(self.templates) == len(self.best_choices), f"Number templates and choices not match, {len(self.templates)}, and {len(self.best_choices)}."
        templates_copies = []
        for i, template in enumerate(self.templates):
            if manual_choice != None:
                self.templates[i].answer_choices = self.possible_choices[manual_choice]
            else:
                self.templates[i].answer_choices = self.possible_choices[self.best_choices[0]]
        
        self.templates += templates_copies
        self.restrict_templates = None

        print("Templates after best choices were applied: ")
        for template in self.templates:
            print(template.jinja)
            print(template.answer_choices)
            print("###########")
        
        with open(os.path.join(save_path, "templates.txt"), "w") as f_out:
            for template in self.templates:
                f_out.write(template.jinja + "\n")
                f_out.write(template.answer_choices + "\n")
                f_out.write("\n")

            
    def apply_choice_to_template(self, choice):
        assert self.current_choice != None, "Need to specify current choice"
        for template in self.templates:
            template.answer_choices = choice


    def apply_next_choice(self):
        if self.current_choice < len(self.possible_choices):
            choice = self.possible_choices[self.current_choice]
            self.apply_choice_to_template(choice)
            self.current_choice +=1
            return 1
        else:
            self.current_choice = 0
            return 0

    def add_possible_choice(self, answer_choices):
        self.possible_choices.insert(0, answer_choices)     


    def restrict_templates_to(self, template_index):
        self.restrict_templates = template_index

    def __init__(self, template_config: dict):
        """
        Initialize the TemplateEngine using preset templates from existing datasets or custom templates specified in config config.data.templates, if specified.

        Parameters
        ---------------
        template_config
            The templates configuration specified in config.data.templates.
        """
        self.templates = []
        self.possible_choices = []
        self.best_choices = []
        self.current_choice = 0
        self.restrict_templates = None
        self.template_config = template_config
        collection = TemplateCollection()
        self.all_datasets = collection.keys
        self.preset_templates = OmegaConf.select(self.template_config, "preset_templates", default=None)
        self.custom_templates = OmegaConf.select(self.template_config, "custom_templates", default=None)
        self.num_templates = OmegaConf.select(self.template_config, "num_templates", default=30)
        self.template_length = OmegaConf.select(self.template_config, "template_length", default=2048)
        self.custom_answer_choices = OmegaConf.select(self.template_config, "custom_answer_choices", default=None)

        self.num_retrieved_templates = OmegaConf.select(self.template_config, "num_retrieved_templates", default=5)
        self.train_data = OmegaConf.select(self.template_config, "train_data", default=None) #train_data.to_dict().values(),

        self.template_choice_mode = OmegaConf.select(self.template_config, "template_choice_mode", default=False)
        self.template_mode = OmegaConf.select(self.template_config, "template_mode", default=False)
        self.template_kb = OmegaConf.select(self.template_config, "template_kb", default=False)
        self.tune_templates = OmegaConf.select(self.template_config, "tune_templates", default=False)


        if self.preset_templates:
            assert (
                len(self.preset_templates) == 2
            ), f"Preset templates has the wrong format. Needs to be [DATASET, SUBSET]."
            dataset_templates = DatasetTemplates(self.preset_templates[0], self.preset_templates[1])
            current_templates = list(dataset_templates.templates.values())

            if self.template_choice_mode == "dataset":
                for template in current_templates:
                    dataset_name = self.map_source_to_dataset(self.preset_templates)
                    template.answer_choices = ' ||| '.join(self.get_dataset_labels(dataset_name)) 
            elif self.template_choice_mode in ["random"]:
                dataset_name = self.map_source_to_dataset(self.preset_templates)
                num_classes = self.get_num_classes(dataset_name)
                vocabulary = set([])
                for entry in self.train_data:
                    vocabulary |= set(entry.split())
                vocabulary = list(vocabulary)
                random_choices = [random.choice(vocabulary) for i in range(num_classes)]
                for template in current_templates:
                    template.answer_choices = " ||| ".join(random_choices)
            elif self.template_choice_mode in ["inverted"]:
                for template in current_templates:
                    choices = template.answer_choices
                    choices = choices.split(" ||| ")
                    choices.reverse()
                    template.answer_choices = " ||| ".join(choices)

            self.templates += current_templates[: self.num_templates]

            if self.preset_templates[1] == 'wsc.fixed': # Some templates throw exception due to "[span2_index:]"
                del self.templates[2:4]

        if self.custom_templates:
            for key, value in self.custom_templates.items():
                if len(self.templates) >= self.num_templates:
                    logger.warning(
                        f"Ignored custom template '{value.template}' as template engine already has {self.num_templates} templates."
                    )
                    break
                if value.answer_choices == "None":
                    value.answer_choices = ' ||| '.join(self.get_best_labels(self.preset_templates[1]))
                template = Template(key, value.template, "custom", answer_choices=value.answer_choices)
                self.templates.append(template)
        
        if not self.has_templates():
            logger.warning(
                f"No preset or custom templates found. Retrieve templates from template database..."
            )
            self.find_templates_from_database(top_k=self.num_retrieved_templates)
            self.generate_possible_choices(use_dataset_choices=self.template_choice_mode=="auto_dataset")

        logger.info(f"Using a total of {len(self.templates)} templates.")

        self.current_counter = random.randint(0,len(self.templates))

        logger.info(f"Starting with template {self.current_counter}.")

    def find_templates_from_database(self, top_k=5):
        from sentence_transformers import SentenceTransformer, util
        import re
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        retrieved_templates = []
        template_candidates = []

        query, query_elements, num_variables = self.get_query_variable_info(self.preset_templates[1])

        num_classes = self.get_num_classes(self.preset_templates[1])

        all_scores = []

        if self.template_kb == "all":
            collection = TemplateCollection()
            all_datasets = collection.keys
            to_exclude = self.map_dataset_to_source(self.preset_templates[1])
            datasets_to_be_considered = all_datasets

            all_datasets.remove(to_exclude)
            if self.preset_templates[1] == 'sst5':
                all_datasets.remove(("SetFit", "sst5"))
        elif self.template_kb == "pretraining":
            datasets_to_be_considered = [["commonsense_qa", ""], ["dream", ""], ["quail", ""], ["social_i_qa", ""], ["wiki_qa", ""], ["cosmos_qa", ""], ["amazon_polarity", ""], ["imdb", ""], ["yelp_polarity", ""], ["rotten_tomatoes", ""], ["ag_news", ""], ["paws", "labeled_final"], ["wiki_qa", ""], ["app_reviews", ""], ["dbpedia_14", ""], ["trec", ""], ["wiki_bio", ""], ["common_gen", ""], ["gigaword", ""], ["multi_news", ""], ["samsum", ""], ["xsum", ""], ["quora", ""], ["quartz", ""], ["wiqa", ""], ["qasc", ""], ["quarel", ""], ["sciq", ""], ["wiki_hop", "original"], ["hotpot_qa", "fullwiki"],
            ]
        for dataset in datasets_to_be_considered:
            dataset_templates = DatasetTemplates(dataset[0], dataset[1])
            current_templates = list(dataset_templates.templates.values())
            for template in current_templates:
                template_fields, answers_in_input, answer_fields = template.get_template_fields()
                adheres_to_format = True
                if len(template_fields) != num_variables:
                    adheres_to_format = False
                elif "%}" in template.jinja: #No macros as heavily task specific
                    adheres_to_format = False
                elif not answer_fields: # No suitable classification output format
                    adheres_to_format = False
                elif len(answer_fields) > 1: #Output label format not for classification
                    adheres_to_format = False
                elif "answer_choices[label]" not in answer_fields and "answer_choices [label]" not in answer_fields: # Need to have label answer_choices in output
                    adheres_to_format = False
                    adheres_to_format = False
                elif answers_in_input: #If output in text ensure that it can be referenced by "label" field.
                    for field in answers_in_input:
                        field_contents = re.findall(r"\[(.+?)\]", field)
                        max_value = -100
                        for field_content in field_contents:
                            field_content = field_content.strip()
                            if not field_content.isnumeric() and field_content != "label":
                                adheres_to_format = False
                            elif field_content.isnumeric():
                                max_value = max(int(field_content), max_value)
                        if max_value != -100 and max_value != num_classes -1: #Check all labels occur in input to avoid bias
                            adheres_to_format = False          

                if adheres_to_format:
                    template_candidates.append(template)

        if self.template_mode == "random":
            random.shuffle(template_candidates)
            retrieved_templates = template_candidates
        elif self.template_mode == "none":
            retrieved_templates = [Template(0, " ".join(["{{" + x + "}}" for x in query_elements]) + "||| {{ answer_choices[label] }}" , "custom", answer_choices = "")]
        else:
            query_enc = model.encode(query, convert_to_tensor=True)
            template_candidates_filtered = []
            template_duplicates = set([])
            for template in template_candidates:
                if template.jinja in template_duplicates:
                    continue
                else:
                    template_duplicates.add(template.jinja)

                doc = str(template.jinja).replace("{", " ").replace("}", " ") 
                doc_enc = model.encode(doc, convert_to_tensor=True)
                score = util.pytorch_cos_sim(query_enc, doc_enc)
                score = score[0][0].item()
                all_scores.append(score)
                template_candidates_filtered.append(template)
            
            if query.isnumeric() or query in ["text", "word", "sentence", "column", "row"]: #Check more generally with regex
                print("Found uninformative data header. Compute contextual representation instead...")
                train_encs = []
                for sample in self.train_data:
                    train_enc = model.encode(sample, convert_to_tensor=True)
                    train_encs.append(train_enc)
                train_encs = torch.stack(train_encs)

                for i, template in enumerate(template_candidates_filtered):
                    doc = str(template.jinja).replace("{", " ").replace("}", " ") 
                    doc_enc = model.encode(doc, convert_to_tensor=True)
                    # doc_enc_repeat = doc_enc.unsqueeze(0).repeat((train_encs.size(0), 1))
                    score = util.pytorch_cos_sim(train_encs, doc_enc)
                    score = torch.sum(score) / train_encs.size(0)
                    all_scores[i] = score.item()


            combined = list(zip(template_candidates_filtered, all_scores))
            combined.sort(key = lambda x: x[1], reverse=True)
            if self.template_mode == "worst_k":
                retrieved_templates = [ele[0] for ele in combined][-top_k:]
            else:
                retrieved_templates = [ele[0] for ele in combined][:top_k]

        for template in retrieved_templates:
            fields, answers_in_input, answer_fields = template.get_template_fields()
            for i, field in enumerate(fields):
                if "answer" in field or "choices" in field:
                    template.jinja = template.jinja.replace("{{" + field + "}}", "{{" + query_elements[i] + "}}")
                else:
                    template.jinja = template.jinja.replace(field, query_elements[i])
                if "choices['label'].index(answerKey)]" in template.jinja or 'choices["label"].index(answerKey)]' in template.jinja:
                    template.jinja = template.jinja.replace("choices['label'].index(answerKey)] ", "label]")
                    template.jinja = template.jinja.replace('choices["label"].index(answerKey)]', "label]")
                    template.jinja = template.jinja.replace('{% if answerKey != "" %}', "").strip()
                    template.jinja = template.jinja.replace("{% endif %}", "").strip()


        for template in retrieved_templates:
            if self.template_choice_mode  in [ "dataset", "auto_dataset"]:
                template.answer_choices = ' ||| '.join(self.get_dataset_labels(self.preset_templates[1]))
            elif self.template_choice_mode in ["auto"]:
                 template.answer_choices = " ||| ".join(self.custom_answer_choices)
            elif self.template_choice_mode in ["random"]:
                vocabulary = set([])
                for entry in self.train_data:
                    vocabulary |= set(entry.split())
                vocabulary = list(vocabulary)
                template.answer_choices = " ||| ".join([random.choice(vocabulary) for i in range(num_classes)])
            
        self.templates += retrieved_templates[:self.num_templates]

        #Remove input choice space
        for template in self.templates:
            replace_field_with = ""
            setattr(template, "jinja", template.jinja.replace("True or False?", replace_field_with).replace("Yes or no?", replace_field_with).replace("Yes or No?", replace_field_with).replace("True or False", replace_field_with).replace("Paraphrase or not?", replace_field_with).replace("Yes or No.\n", replace_field_with))

        for template in self.templates:
            print(template.jinja)
            print(template.answer_choices)
            print("###########")


    def has_templates(self):
        return len(self.templates) > 0

    def get_templates(self):
        return self.templates

    def sample_and_apply_template_all(self, example: dict):
        if not self.templates:
            return [None, example]   
        all_applied = []
        all_templates = []
        for i in range(len(self.templates)):
            if self.restrict_templates != None:
                template = self.templates[self.restrict_templates]
            else:
                template = self.templates[i]
            all_applied.append(template.apply(example, truncation_length=self.template_length))
            all_templates.append(template)
        
        # print(all_applied)

        assert len(self.templates) == len(all_applied), f"Not same length {len(self.templates)}, {len(all_applied)}"
        assert len(all_templates) == len(all_applied)
        return [all_templates, all_applied]

        
    def sample_and_apply_template(self, example: dict):
        """
        Randomly sample a template from the collection of available templates and apply it to the sample.
        If collection of templates is empty return original sample.

        Parameters
        ---------------
        example
            A data sample, i.e. a dictionary of text columns.

        Returns
        ------------------
        A tuple consisting of the selected tuple and the sample after the template has been applied to it.
        """
        if not self.templates:
            return [None, example]

        # self.templates = self.templates[:3] + self.templates[4:]
        # self.templates = self.templates[:2]
        if self.restrict_templates != None:
            template = self.templates[self.restrict_templates]
        else:
            template = np.random.choice(self.templates)

        # template = self.templates[2]
        # print(self.templates.index(template))
        return [template, template.apply(example, truncation_length=self.template_length)]


    ########TOODO FROM HERE: HARDCODED CURRENTLY, BAD STYLE. FIX BY READING IN DATASET (OR PASSING ARUGMENTS THROUGH AUTOGLUON)

    def get_num_classes(self, dataset_name):
        #TODO: Hardcoded, read instead directly from dataset
        if dataset_name == "rte":
            return 2
        elif dataset_name == "wic":
            return 2
        elif dataset_name == "wsc":
            return 2
        elif dataset_name == "cb":
            return 3
        elif "anli" in dataset_name:
            return 3
        elif dataset_name == "emotion":
            return 6
        elif dataset_name == "sst5":
            return 5
        elif dataset_name == "amazon_counterfactual_en":
            return 2
        elif dataset_name == "enron_spam":
            return 2
        elif dataset_name == "sentevalcr":
            return 2


    def generate_possible_choices(self, use_dataset_choices=False):
        # TODO: Currently hard-coded because lazy. Should be read from dataset directly!
        self.possible_choices.append(" ||| ".join(self.custom_answer_choices))
        if use_dataset_choices:
            if self.preset_templates[1] in ['rte']:
                self.possible_choices.append("entailment ||| not_entailment")
            elif self.preset_templates[1] in ["wic", "wsc"]:
                self.possible_choices.append("No ||| Yes")
            elif self.preset_templates[1] in ['anli-r1', 'anli-r2', 'anli-r3']:
                self.possible_choices.append("entailment ||| neutral ||| contradiction")
            elif self.preset_templates[1] in ['cb']:
                self.possible_choices.append("entailment ||| contradiction ||| neutral")
            elif self.preset_templates[1] in ["sentevalcr", "emotion", "enron_spam", "sst5", "amazon_counterfactual_en"]:
                if self.preset_templates[1] == "emotion":
                    answer_choices = ["sadness", "joy", "love", "anger", "fear", "surprise"]
                elif self.preset_templates[1] == "enron_spam":
                    answer_choices = ["ham", "spam"]
                elif self.preset_templates[1] == "sentevalcr":
                    answer_choices = ["negative", "positive"]
                elif self.preset_templates[1] == 'sst5':
                    answer_choices = ["very negative", "negative", "neutral", "positive", "very positive"]
                elif self.preset_templates[1] == 'amazon_counterfactual_en':
                    answer_choices = ["not-counterfactual", "counterfactual"]
                self.possible_choices.append(' ||| '.join(answer_choices))          

    def get_dataset_labels(self, dataset_name):
        if dataset_name in ['rte']:
            return ["entailment", "not_entailment"]
        elif dataset_name in ["wic", "wsc", "wsc.fixed"]:
            return ["No", "Yes"]
        elif dataset_name in ['anli-r1', 'anli-r2', 'anli-r3']:
            return ["entailment" , "neutral", "contradiction"]
        elif dataset_name in ['cb']:
            return ["entailment", "neutral", "contradiction"]
        elif dataset_name in ["sentevalcr", "emotion", "enron_spam", "sst5", "amazon_counterfactual_en"]:
            if dataset_name == "emotion":
                return ["sadness", "joy", "love", "anger", "fear", "surprise"]
            elif dataset_name == "enron_spam":
                return ["ham", "spam"]
            elif dataset_name == "sentevalcr":
                return ["negative", "positive"]
            elif dataset_name == 'sst5':
                return ["very negative", "negative", "neutral", "positive", "very positive"]
            elif dataset_name == 'amazon_counterfactual_en':
                return ["not-counterfactual", "counterfactual"]

    def map_source_to_dataset(self, dataset_name):
            if dataset_name == ["super_glue", "rte"]:
                return "rte"
            if dataset_name == ["super_glue", "wic"]:
                return "wic"
            if dataset_name == ["super_glue", "wsc.fixed"]:
                return "wsc"
            if dataset_name == ["super_glue", "cb"]:
                return "cb"
            if dataset_name == ["anli-r1", ""]:
                return "anli-r1"
            if dataset_name == ["anli-r2", ""]:
                return "anli-r2"
            if dataset_name == ["anli-r3", ""]:
                return "anli-r3"
            if dataset_name == ["emotion", ""]:
                return "emotion"
            if dataset_name == ["SetFit", "sst5"]:
                return "sst5"
            if dataset_name == ["SetFit", "SentEval-CR"]:
                return "sentevalcr"
            if dataset_name == ["SetFit", "enron_spam"]:
                return "enron_spam"
            if dataset_name == ["SetFit", "amazon_counterfactual_en"]:
                return "amazon_counterfactual_en"

    def map_dataset_to_source(self, dataset_name):
            if dataset_name == "rte":
                return ("super_glue", "rte")
            elif dataset_name == "wic":
                return ("super_glue", "wic")
            elif dataset_name == "wsc":
                return ("super_glue", "wsc.fixed")
            elif dataset_name == "cb":
                return ("super_glue", "cb")
            elif "anli" in dataset_name:
                return ("anli", None)
            elif dataset_name == "emotion":
                return ("emotion", None)
            elif dataset_name == "sst5":
                return ("sst", "default")
            elif dataset_name == "amazon_counterfactual_en":
                return ("SetFit", "amazon_counterfactual_en")
            elif dataset_name == "enron_spam":
                return ("SetFit", "enron_spam")
            elif dataset_name == "sentevalcr":
                return ("SetFit", "SentEval-CR")

    def get_query_variable_info(self, dataset_name):
        if dataset_name in ['wic']:
            query = "sentence1 and sentence2 and word"
            query_elements = ["sentence1", "sentence2", "word"]
            num_variables = 3
        elif dataset_name in ['wsc']:
            query = "text and span1 text and span2 text"
            query_elements = ["text", "span1_text", "span2_text"]
            num_variables = 3
        elif dataset_name in ["sentevalcr", "emotion", "enron_spam", "sst5", "amazon_counterfactual_en"]: # Note all datasets that SetFit is evaluated on are simple in structure
            query = "text"
            query_elements = ["text"]
            num_variables = 1
        else:    # rte and cb and wsc
            query = "premise and hypothesis"
            query_elements = ["premise", "hypothesis"]
            num_variables = 2
        
        return [query, query_elements, num_variables]

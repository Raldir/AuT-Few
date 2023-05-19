# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import numpy as np
import random


def identity(x):
    return x


class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LinearEnsembleExperimental(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_ensemble: int,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=True)  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        # self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False
        self.current_template = 0
        # self.current_sample = 0

        self.num_templates = -1
        # self.template_to_ensemble = {0: [0, 1], 1: [0, 2], 2: [0, 3], 3: [0, 4], 4: [1,2], 5: [1,3], 6: [1,4], 7: [2, 3], 8: [2,4], 9: [3, 4]}
        self.num_total_ensemble =  num_ensemble
        all_template_weights = [nn.Parameter(torch.ones(out_features, 1)) for i in range(self.num_total_ensemble)]
        # print(all_template_weights)
        self.lora_b = nn.ParameterList(all_template_weights)
        # self.every_n = 0

        # for i, l in enumerate(self.lora_b):
        #     if i == 0 or i >=3:
        #         continue
        #     print(self.lora_b[i])
        #     self.lora_b[0] = (self.lora_b[0] + self.lora_b[i])
        # self.lora_b[0] = self.lora_b[0] / 3 # len(self.lora_b)

            # self.lora_b_souped = (self.lora_b_souped + self.lora_b[i]) / 2.

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        # hidden = hidden * self.lora_b[.flatten()

        # current_template = self.current_template.item()
        # hidden_results = []
        # for index in self.template_to_ensemble[current_template]:
        #     hidden0 = hidden * self.lora_b[index].flatten()
        #     hidden_results.append(hidden0)

        # hidden_sum = hidden_results[0]
        # for i, rep in enumerate(hidden_results):
        #     if i == 0:
        #         continue
        #     hidden_sum += rep  
        # hidden = hidden_sum/len(hidden_results)


        if self.training:
            num_templates_per_ensemble = math.ceil(self.num_templates/self.num_total_ensemble)
            target_ensemble = int(self.current_template/num_templates_per_ensemble)
            hidden = hidden * self.lora_b[target_ensemble].flatten()

            # if self.current_template > int(self.num_templates/self.num_total_ensemble):
            #     hidden = hidden * self.lora_b[0].flatten()
            # else:
            #     hidden = hidden * self.lora_b[1].flatten()
            # hidden = hidden * self.lora_b[self.current_template.item().flatten()
        else:
            # for i, l in enumerate(self.lora_b):
            #     if i == 0 or i >=3:
            #         continue
            #     self.lora_b[0] = (self.lora_b[0] + self.lora_b[i])
            # self.lora_b[0] = self.lora_b[0] / 3 # len(self.lora_b)
            # hidden = hidden * self.lora_b[0].flatten()
            # self.every_n +=1
            # if self.every_n % 200 == 0:
            #     print(self.lora_b[7])
            #     print("_____________")

            hidden_results = []
            for i, l in enumerate(self.lora_b):
                hidden0 = hidden * self.lora_b[i].flatten()
                hidden_results.append(hidden0)
            
            hidden_sum = hidden_results[0]
            for i, rep in enumerate(hidden_results):
                if i == 0:
                    continue
                hidden_sum += rep  
            hidden = hidden_sum/len(hidden_results)

        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

class IA3LinearEnsemble(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_ensemble: int,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=True)  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        # self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False
        self.num_total_ensemble = num_ensemble #equal to number of seeds
        self.current_ensemble_iteration = 0
        self.all_parameters = []
        for i in range(self.num_total_ensemble):
            setattr(self, "lora_b_old_" + str(i), nn.Parameter(torch.ones(out_features, 1)))
            self.all_parameters.append(getattr(self, "lora_b_old_" + str(i)))
        
        self.current_step = 0

    def enable_ensemble_weights(self):
        # all_template_weights = [copy.deepcopy(self.lora_b_old) for i in range(self.num_total_ensemble)]
        self.lora_b = nn.ParameterList(self.all_parameters)

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)

        # self.current_step+=1
        # if self.current_step % 2 == 0:
        #     print(self.lora_b[0])
        #     print(self.lora_b[1])
        #     print("-----------------")
        
        hidden = hidden * self.lora_b[self.current_ensemble_iteration].flatten()

        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=True)  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False
        self.current_template = 0
        # print(all_template_weights)
        # self.lora_b = nn.ParameterList(all_template_weights)
        # self.every_n = 0

        # for i, l in enumerate(self.lora_b):
        #     if i == 0 or i >=3:
        #         continue
        #     print(self.lora_b[i])
        #     self.lora_b[0] = (self.lora_b[0] + self.lora_b[i])
        # self.lora_b[0] = self.lora_b[0] / 3 # len(self.lora_b)

            # self.lora_b_souped = (self.lora_b_souped + self.lora_b[i]) / 2.

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()

        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r = 8,
        lora_alpha = 8,
        merge_weights = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=8, lora_dropout=0.0, merge_weights=False)  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.weight.requires_grad = False

        self.fan_in_fan_out = False

        self.current_template = 0
        self.current_ensemble_iteration = 0
        # all_template_weights = [nn.Parameter(torch.empty(out_features, 1).normal_(mean=1, std=0.1)) for i in range(self.num_total_ensemble)]
        # print(all_template_weights)

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()

        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3LoRAEnsembleLinear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r = 8,
        lora_alpha = 8,
        merge_weights = False,
        num_ensemble = 1,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=8, lora_dropout=0.0, merge_weights=False)  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.all_parameters = []
        self.all_parameters_A = []
        self.all_parameters_B = []
        
        for i in range(1):
            setattr(self, "lora_A_old_" + str(i), nn.Parameter(self.weight.new_zeros((r, in_features))))
            setattr(self, "lora_B_old_" + str(i), nn.Parameter(self.weight.new_zeros((out_features, r))))
            self.all_parameters_A.append(getattr(self, "lora_A_old_" + str(i)))
            self.all_parameters_B.append(getattr(self, "lora_B_old_" + str(i)))

        # self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        # self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.weight.requires_grad = False

        self.fan_in_fan_out = False

        self.current_template = 0
        self.num_total_ensemble = num_ensemble #equal to number of seeds
        self.current_ensemble_iteration = 0
        # all_template_weights = [nn.Parameter(torch.empty(out_features, 1).normal_(mean=1, std=0.1)) for i in range(self.num_total_ensemble)]
        # print(all_template_weights)

        for i in range(self.num_total_ensemble-1):
            setattr(self, "lora_b_old_" + str(i), nn.Parameter(torch.ones(out_features, 1)))
            self.all_parameters.append(getattr(self, "lora_b_old_" + str(i)))
        
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()
        self.current_step = 0

    def enable_ensemble_weights(self):
        # all_template_weights = [copy.deepcopy(self.lora_b_old) for i in range(self.num_total_ensemble)]
        self.lora_b = nn.ParameterList(self.all_parameters)
        self.lora_A = nn.ParameterList(self.all_parameters_A)
        self.lora_B = nn.ParameterList(self.all_parameters_B)


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            for element in self.all_parameters_A:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(element, a=math.sqrt(5))
            for element in self.all_parameters_B:
                nn.init.zeros_(element)


    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        if self.current_ensemble_iteration == 0:
            if self.r > 0 and not self.merged:
                result = F.linear(x, self.T(self.weight), bias=self.bias)
                if self.r > 0:
                    result += (self.lora_dropout(x) @ self.lora_A[self.current_ensemble_iteration].T @ self.lora_B[self.current_ensemble_iteration].T) * self.scaling
                return result
            else:
                return F.linear(x, self.T(self.weight), bias=self.bias) 
        else:
            hidden = F.linear(x, self.weight, self.bias)
            hidden = hidden * self.lora_b[self.current_ensemble_iteration -1].flatten()

        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinearEnsemble(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        num_ensemble: int = 1,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        self.current_template = 0
        self.num_total_ensemble = num_ensemble #equal to number of seeds
        self.current_ensemble_iteration = 0
        # all_template_weights = [nn.Parameter(torch.empty(out_features, 1).normal_(mean=1, std=0.1)) for i in range(self.num_total_ensemble)]
        # print(all_template_weights)

        # Actual trainable parameters
        if r > 0:
            all_template_weights_a = [nn.Parameter(self.weight.new_zeros((r, in_features))) for i in range(self.num_total_ensemble)]
            all_template_weights_b = [nn.Parameter(self.weight.new_zeros((out_features, r))) for i in range(self.num_total_ensemble)]

            self.lora_A = nn.ParameterList(all_template_weights_a) # nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.ParameterList(all_template_weights_b) #nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for content in self.lora_A:
                nn.init.kaiming_uniform_(content, a=math.sqrt(5))
            for content in self.lora_B:
                nn.init.zeros_(content)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A[self.current_ensemble_iteration].T @ self.lora_B[self.current_ensemble_iteration].T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)

class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)

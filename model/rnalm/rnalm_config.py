# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT model configuration """
from transformers.models.esm.configuration_esm import EsmConfig as TransformersEsmConfig


class RNALMConfig(TransformersEsmConfig):

    def __init__(
        self,
        alibi_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,
        use_ALiBi=None,
        use_flash_attn=None,
        token_type=None,
        **kwargs,
    ):
        """Configuration class for MosaicBert.
        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.alibi_starting_size = alibi_starting_size
        self.use_ALiBi =use_ALiBi
        self.use_flash_attn =use_flash_attn
        self.token_type = token_type



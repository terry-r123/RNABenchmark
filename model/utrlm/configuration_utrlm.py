# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers.utils import logging

from ..configuration_utils import HeadConfig, MaskedLMHeadConfig, PretrainedConfig

logger = logging.get_logger(__name__)


class UtrLmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UtrLmModel`]. It is used to instantiate a RNA-FM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RNA-FM
    [a96123155/UTR-LM](https://github.com/a96123155/UTR-LM) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 25):
            Vocabulary size of the RNA-FM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`UtrLmModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the RnaBert code use this instead of the attention mask.
        bos_token_id (`int`, *optional*, defaults to 1):
            The index of the bos token in the vocabulary. This must be included in the config because of the
            contact and other prediction heads removes the bos and padding token when predicting outputs.
        mask_token_id (`int`, *optional*, defaults to 4):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.

    Examples:
        >>> from multimolecule import UtrLmModel, UtrLmConfig

        >>> # Initializing a UTR-LM multimolecule/utrlm style configuration
        >>> configuration = UtrLmConfig()

        >>> # Initializing a model (with random weights) from the multimolecule/utrlm style configuration
        >>> model = UtrLmModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "utrlm"

    def __init__(
        self,
        vocab_size=25,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="rotary",
        use_cache=True,
        emb_layer_norm_before=None,
        token_dropout=True,
        head=None,
        lm_head=None,
        structure_head=None,
        supervised_head=None,
        token_type=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.head = HeadConfig(**head if head is not None else {})
        self.lm_head = MaskedLMHeadConfig(**lm_head if lm_head is not None else {})
        self.structure_head = HeadConfig(**structure_head) if structure_head is not None else None
        self.supervised_head = HeadConfig(**supervised_head) if supervised_head is not None else None
        self.token_type = token_type


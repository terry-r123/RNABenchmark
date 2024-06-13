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


class RnaMsmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RnaMsmModel`]. It is used to instantiate a
    RnaMsm model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RnaMsm
    [yikunpku/RNA-MSM](https://github.com/yikunpku/RNA-MSM) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 25):
            Vocabulary size of the RnaMsm model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RnaMsmModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 10):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Examples:
        >>> from multimolecule import RnaMsmModel, RnaMsmConfig

        >>> # Initializing a RNA-MSM multimolecule/rnamsm style configuration
        >>> configuration = RnaMsmConfig()

        >>> # Initializing a model (with random weights) from the multimolecule/rnamsm style configuration
        >>> model = RnaMsmModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "rnamsm"

    def __init__(
        self,
        vocab_size=25,
        hidden_size=768,
        num_hidden_layers=10,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        max_tokens_per_msa=2**14,
        attention_type="standard",
        embed_positions_msa=True,
        attention_bias=True,
        head=None,
        lm_head=None,
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
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attention_type = attention_type
        self.embed_positions_msa = embed_positions_msa
        self.attention_bias = attention_bias
        self.head = HeadConfig(**head if head is not None else {})
        self.lm_head = MaskedLMHeadConfig(**lm_head if lm_head is not None else {})
        self.token_type = token_type

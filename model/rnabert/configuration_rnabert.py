from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class RnaBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RnaBertModel`]. It is used to instantiate a
    RnaBert model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RnaBert
    [mana438/RNABERT](https://github.com/mana438/RNABERT) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the RnaBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RnaBertModel`].
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the RnaBert code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Examples:

    ```python
    >>> from multimolecule import RnaBertModel, RnaBertConfig

    >>> # Initializing a RnaBert style configuration >>> configuration = RnaBertConfig()

    >>> # Initializing a model from the configuration >>> model = RnaBertModel(configuration)

    >>> # Accessing the model configuration >>> configuration = model.config
    ```
    """

    model_type = "rnabert"

    def __init__(
        self,
        vocab_size=25,
        ss_vocab_size=8,
        hidden_size=None,
        multiple=None,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=40,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=440,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.ss_vocab_size = ss_vocab_size
        if hidden_size is None:
            hidden_size = num_attention_heads * multiple if multiple is not None else 120
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

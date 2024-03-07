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
"""PyTorch BERT model. """


import logging
import math
import os
import pdb
import torch
from torch import nn, einsum
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.autograd as autograd


from transformers import PreTrainedModel
from model.rnalm_config import RNALMConfig as BertConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.activations import gelu, gelu_new
#from .modeling_utils import , 
#import List
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu,  "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm

### ALiBi modules ###

def rebuild_alibi_tensor(    num_attention_heads: int,
                             size: int,
                             device: Optional[Union[torch.device, str]] = None):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:

            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = (2**(-2**-(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][:n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(
            n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])
        return alibi
        
###
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        # try:
        #     config.use_ALiBi
        # except:
        #     config.use_ALiBi = False
        # try:
        #     config.use_Enformer_pos
        # except:
        #     config.use_Enformer_pos = False
        #assert not (config.use_ALiBi and config.use_Enformer_pos), "ALiBi and Enformer positional embeddings cannot be used together"
        print(f'position embedding type is {self.position_embedding_type}')
        if self.position_embedding_type == 'alibi' or self.position_embedding_type == 'rope' or self.position_embedding_type == 'nope':
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if self.position_embeddings is None:
            position_embeddings = torch.zeros_like(inputs_embeds)
        else:
            position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)
        # enformer relative positional encoding
        try:
            self.use_Enformer_pos = config.use_Enformer_pos
        except:
            self.use_Enformer_pos = False
        if self.use_Enformer_pos:
            self.num_rel_pos_features = config.hidden_size // self.num_attention_heads
            self.to_rel_k = nn.Linear(self.num_rel_pos_features,self.all_head_size, bias = False)
            self.rel_content_bias = nn.Parameter(torch.randn(1, self.num_attention_heads, 1, self.attention_head_size))
            self.rel_pos_bias = nn.Parameter(torch.randn(1, self.num_attention_heads, 1, self.attention_head_size))
            self.pos_dropout = nn.Dropout(config.enformer_posdropout)
        else:
            self.num_rel_pos_features = 0

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.config = config
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.position_embedding_type == "rotary":
            query_layer = query_layer * self.attention_head_size**-0.5
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        
        if self.use_Enformer_pos:
            from enformer_positional_embeddings import get_positional_embed, relative_shift
            n, h, device = hidden_states.shape[-2], self.num_attention_heads, hidden_states.device
            content_logits = einsum('b h i d, b h j d -> b h i j', query_layer + self.rel_content_bias, key_layer)
            positions = get_positional_embed(n, self.num_rel_pos_features, device)
            positions = self.pos_dropout(positions)
            rel_k = self.to_rel_k(positions)
            
            rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
            rel_logits = einsum('b h i d, h j d -> b h i j', query_layer + self.rel_pos_bias, rel_k)
            rel_logits = relative_shift(rel_logits)
            
            logits = content_logits + rel_logits
            attention_scores = logits / math.sqrt(self.attention_head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)         
            # print(attention_mask[0][0:2])
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        #print(self.config)
        if  self.config.fpmix == 'fp16':
            context_layer = torch.matmul(attention_probs.float(), value_layer.float()).half()
        else:
            context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
    #     heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
    #     for head in heads:
    #         # Compute how many pruned heads are before the head and move the index accordingly
    #         head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
    #         mask[head] = 0
    #     mask = mask.view(-1).contiguous().eq(1)
    #     index = torch.arange(len(mask))[mask].long()

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        # print(hidden_states.shape)
        # print(attention_mask.shape)
        # print(head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


### adding layer scale to bertlayer ###

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
        # return x.mul_(self.gamma.half()) if self.inplace else x * self.gamma.half() 
        # return x.float().mul_(self.gamma).half() if self.inplace else (x.float() * self.gamma).half()

class BertSelfOutput_lscale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerscale = LayerScale(config.hidden_size, config.lscale_init_values)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layerscale(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention_lscale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput_lscale(config)
        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
    #     heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
    #     for head in heads:
    #         # Compute how many pruned heads are before the head and move the index accordingly
    #         head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
    #         mask[head] = 0
    #     mask = mask.view(-1).contiguous().eq(1)
    #     index = torch.arange(len(mask))[mask].long()

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        # print(hidden_states.shape)
        # print(attention_mask.shape)
        # print(head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertOutput_lscale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerscale = LayerScale(config.hidden_size, config.lscale_init_values)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layerscale(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer_lscale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention_lscale(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention_lscale(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput_lscale(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # pdb.set_trace()

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
  
### adding layer scale to bertlayer ###

### Using flash attention

#same as  class FlashMHA(nn.Module):

import math
from einops import rearrange
# Try to import flash_attn, if not found, assign None
# from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
# from flash_attn.bert_padding import unpad_input, pad_input
# from flash_attn.flash_attention import FlashAttention
try :
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    # from flash_attn.flash_attention import FlashAttention
except:
    flash_attn_unpadded_qkvpacked_func = None
    unpad_input = None
    pad_input = None
    # FlashAttention = None
    

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                # print(qkv.shape)
                # print(cu_seqlens.shape)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                # print(qkv.shape)
                # print(cu_seqlens.shape)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class BertSelfAttention_flashattn(nn.Module):
    def __init__(self, config,embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        # contains: Linear(hidden->qkv), Attention, Linear
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)
    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        qkv = self.Wqkv(x)
        # print(x.shape)
        # print(qkv.shape)
        # print(key_padding_mask.shape)
        key_padding_mask = key_padding_mask.squeeze()
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        # print(qkv.shape)
        # original_dtype = qkv.dtype
        # qkv = qkv.half()
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        # convert back to original dtype
        # context = context.to(original_dtype)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights

class BertSelfOutput_flashattn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention_flashattn(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        attn_dropout = config.attention_probs_dropout_prob
        self.self = BertSelfAttention_flashattn(config,embed_dim,num_heads,attention_dropout = attn_dropout)
        self.output = BertSelfOutput_flashattn(config)
        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
    #     heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
    #     for head in heads:
    #         # Compute how many pruned heads are before the head and move the index accordingly
    #         head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
    #         mask[head] = 0
    #     mask = mask.view(-1).contiguous().eq(1)
    #     index = torch.arange(len(mask))[mask].long()

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # if attention_mask is not None:
        #     print(attention_mask.shape)
        # else:
        #     print("None attn")
        # if head_mask is not None:
        #     print(head_mask.shape)
        # else:
        #     print("None head")
        # if encoder_hidden_states is not None:
        #     print(encoder_hidden_states.shape)
        # else:
        #     print("None hidden")
        # if encoder_attention_mask is not None:
        #     print(encoder_attention_mask.shape)
        # else:
        #     print("None encoder_attn")
        
        self_outputs = self.self(
            hidden_states, 
            attention_mask,
            #  head_mask, 
            # encoder_hidden_states, encoder_attention_mask
        )
        # print(hidden_states.shape)
        # print(attention_mask.shape)
        # print(head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertLayer_flashattn(nn.Module):
    def __init__(self, config):
        super().__init__()
        print('------------')
        print('using flash attn!!!')
        print('------------')
        self.attention = BertAttention_flashattn(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

### Using flash attention

### Use longnet Dilated Attention （Under Active Development）

import torch 
import torch.nn as nn
import torch.nn.functional as Fdropout



#add alibi, qk layer norm, one write head, multihway, 
class BertSelfOutput_longnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertAttention_longnet(nn.Module):
    def __init__(self, config):
        #d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False
        super(BertAttention_longnet, self).__init__()
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attn_dropout = config.attention_probs_dropout_prob

        self.dilation_rate = config.dilation_rate
        self.segment_size = config.segment_size

        self.attention = BertSelfAttention_flashattn(self.d_model,self.num_heads,attention_dropout = self.attn_dropout)
        self.output = BertSelfOutput_longnet(config)
        
    def forward(self, x,attention_mask,head_mask):
        batch_size, seq_len, _ = x.shape
        # if self.use_xpos:
        #     x = self.xpos(x)
        # Split and sparsify
        #save x for residual connection into hidden_states
        hidden_states = x
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        x = x[:, :, :: self.dilation_rate, :]

        # Perform attention
        attn_output, _ = self.attention(x, x, x)

        #if use rel pos => apply relative positioning bias 
        # if self.use_rel_pos_bias:
        #     attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

        # # if casual create a mask and apply to the output
        # if self.casual:
        #     mask = self.get_mask(attn_output.size(1), attn_output.size(1))
        #     attn_output = attn_output.masked_fill(mask, float('-inf'))
        # Scatter and concatenate 
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        attention_output = self.output(attn_output, hidden_states)
        outputs = (attention_output,) + _
        return attn_output
    
class BertLayer_longnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention_longnet(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

###

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_attention_heads = config.num_attention_heads
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        try:
            config.use_ALiBi
        except:
            config.use_ALiBi = False
        self.use_ALiBi = config.use_ALiBi
        if self.position_embedding_type == 'alibi':
            self.current_ALiBi_size = int(config.alibi_starting_size)
            self.ALiBi_bias = torch.zeros(
            (1, self.num_attention_heads, self.current_ALiBi_size,
             self.current_ALiBi_size))
            self.ALiBi_bias = rebuild_alibi_tensor(self.num_attention_heads,self.current_ALiBi_size)
        
        # pdb.set_trace()
        if getattr(config, 'lscale_init_values', False):
            self.layer = nn.ModuleList([BertLayer_lscale(config) for _ in range(config.num_hidden_layers)])
        elif getattr(config, 'use_flash_attn', False):
            self.layer = nn.ModuleList([BertLayer_flashattn(config) for _ in range(config.num_hidden_layers)])
        elif getattr(config, 'use_longnet', False):
            self.layer = nn.ModuleList([BertLayer_longnet(config) for _ in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        batch, seqlen = hidden_states.shape[:2]
        # check if ALiBi_bias is one the same device as hidden_states
        if self.position_embedding_type == 'alibi':
            if self.ALiBi_bias.device != hidden_states.device:
                self.ALiBi_bias = self.ALiBi_bias.to(hidden_states.device)
                
            if self.current_ALiBi_size < seqlen:
            # Rebuild the alibi tensor when needed
                self.current_ALiBi_size = seqlen
                self.ALiBi_bias = rebuild_alibi_tensor(self.num_attention_heads,self.current_ALiBi_size,device=hidden_states.device)

                
            #  extend the attention mask to include the ALiBi bias
            # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            alibi_bias = self.ALiBi_bias[:, :, :seqlen, :seqlen]
            attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
            alibi_attn_mask = attn_bias + alibi_bias
            attention_mask = alibi_attn_mask
            
            
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]
            # pdb.set_trace()

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        #pdb.set_trace()
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            
            for param in module.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_normal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        #print(encoder_hidden_states)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#        # ourselves in which case we just need to make it broadcastable to all heads.
#        if attention_mask.dim() == 3:
#            extended_attention_mask = attention_mask[:, None, :, :]
#        elif attention_mask.dim() == 2:
#            # Provided a padding mask of dimensions [batch_size, seq_length]
#            # - if the model is a decoder, apply a causal mask in addition to the padding mask
#            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
#            if self.config.is_decoder:
#                batch_size, seq_length = input_shape
#                seq_ids = torch.arange(seq_length, device=device)
#                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
#                causal_mask = causal_mask.to(
#                    attention_mask.dtype
#                )  # causal and attention masks must have same type with pytorch version < 1.3
#                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
#            else:
#                extended_attention_mask = attention_mask[:, None, None, :]
#        else:
#            raise ValueError(
#                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
#                    input_shape, attention_mask.shape
#                )
#            )
#
#        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#        # masked positions, this operation will create a tensor which is 0.0 for
#        # positions we want to attend and -10000.0 for masked positions.
#        # Since we are adding it to the raw scores before the softmax, this is
#        # effectively the same as removing these entirely.
#        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extened_attention_mask = self.invert_attention_mask(encoder_attention_mask)

#            if encoder_attention_mask.dim() == 3:
#                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
#            elif encoder_attention_mask.dim() == 2:
#                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
#            else:
#                raise ValueError(
#                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
#                        encoder_hidden_shape, encoder_attention_mask.shape
#                    )
#                )
#
#            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
#                dtype=next(self.parameters()).dtype
#            )  # fp16 compatibility
#            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None


        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#        if head_mask is not None:
#            if head_mask.dim() == 1:
#                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
#            elif head_mask.dim() == 2:
#                head_mask = (
#                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#                )  # We can specify head_mask for each layer
#            head_mask = head_mask.to(
#                dtype=next(self.parameters()).dtype
#            )  # switch to fload if need + fp16 compatibility
#        else:
#            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        #if not encoder_hidden_states:
        sequence_output = encoder_outputs[0]
        # else:
        #     sequence_output = None
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        next_sentence_label=None,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.


    Examples::

        from transformers import BertTokenizer, BertForPreTraining
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        prediction_scores, seq_relationship_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """
        #pdb.set_trace()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)




@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     #checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
    #     expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            #print(self.config.problem_type)
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        #print('return_dict',return_dict)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

class BertForSequenceRNAdegra(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # if self.use_features:
        #     self.classifier = nn.Linear(config.hidden_size +8 , config.num_labels)
        # else:
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # weight_mask: Optional[bool] = None,
        # post_token_length: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if self.use_features:
        #     features = input_ids[:,-8:]
        #     input_ids = input_ids[:,:-8]
        # else:
        #     input_ids = input_ids
        #     # features = None
        # # pdb.set_trace
        # input_ids = input_ids.int()
        #print(input_ids.shape)
        total_mask_dim = attention_mask.shape[1]
        if self.config.token_type == '6mer':
            cur_length = int((total_mask_dim-5)/3)
            attention_mask, weight_mask, post_token_length = torch.split(attention_mask, (cur_length,cur_length+5,cur_length), dim=1)
        else:
            cur_length = int(total_mask_dim/3)
            attention_mask, weight_mask, post_token_length = torch.split(attention_mask, (cur_length,cur_length,cur_length), dim=1)
        #print(cur_length)
        #attention_mask, weight_mask, post_token_length = torch.split(attention_mask, (cur_length,cur_length+5,cur_length), dim=1)
        #print('---------------------',attention_mask.shape,weight_mask.shape,post_token_length.shape)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        #print(outputs[0].shape)
        final_input= outputs[0]

        hidden_states = self.dropout(final_input)
        ### init mappint tensor
        # ori_length = weight_mask.shape[1]
        # batch_size = final_input.shape[0]
        # #cur_length = final_input.shape[1]
        # assert cur_length == final_input.shape[1]
        # padding_tensor = torch.zeros((batch_size, ori_length-cur_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
        # mapping_final_input =torch.cat([padding_tensor, final_input], dim=1)
        # mapping_final_input[:,0,:] = final_input[:,0,:] #[cls] token
        batch_size = hidden_states.shape[0]
        
        ori_length = weight_mask.shape[1]
        assert cur_length == hidden_states.shape[1]
        padding_tensor = torch.zeros((batch_size, ori_length-cur_length, hidden_states.shape[-1]),dtype=hidden_states.dtype,device=hidden_states.device)
        mapping_final_input = torch.cat([padding_tensor, hidden_states],dim=1)
        mapping_final_input[:,0,:] = hidden_states[:,0,:] # [cls]
        if self.config.token_type == 'bpe':
            for bz in range(batch_size):
                valid_length = torch.sum(attention_mask[bz,:]==1).item()
                for i in range(1, valid_length-1):
                    mapping_final_input[bz,i:i+6,:] += hidden_states[bz,i,:]
                mapping_final_input[bz,valid_length+5-1,:] = hidden_states[bz,valid_length-1,:] #[sep]
        elif self.config.token_type == '6mer':
            for bz in range(batch_size):
                valid_length = torch.sum(attention_mask[bz,:]==1).item()
                for i in range(1, valid_length-1):
                    mapping_final_input[bz,i:i+6,:] += hidden_states[bz,i,:]
                mapping_final_input[bz,valid_length+5-1,:] = hidden_states[bz,valid_length-1,:] #[sep]
        #print(mapping_final_input.shape,weight_mask.shape)
        #print(mapping_final_input,weight_mask)
        mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
        #print(mapping_final_input)
        logits = self.classifier(mapping_final_input)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MCRMSELoss()
                #loss_fct = nn.MSELoss()
                #print('model',logits.shape,labels.shape)
                logits = logits[:, 1:1+labels.size(1), :]
                #loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                    # print(loss)
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            #print('xxxxxxxxxxxxxxxxxxx')
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        #print('yyyyyyyyyyyyyy')
        #print(logits.shape)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
        return score
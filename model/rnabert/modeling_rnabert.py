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
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from warnings import warn
import math
import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.autograd as autograd
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging

from .configuration_rnabert import RnaBertConfig

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
        
class RnaBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RnaBertConfig
    base_model_prefix = "rnabert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RnaBertLayer", "RnaBertEmbeddings"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RnaBertModel(RnaBertPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaBertConfig, RnaBertModel, RnaTokenizer
        >>> config = RnaBertConfig()
        >>> model = RnaBertModel(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
    """

    def __init__(self, config: RnaBertConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.embeddings = RnaBertEmbeddings(config)
        self.encoder = RnaBertEncoder(config)
        self.pooler = RnaBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | torch.Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        
        if attention_mask is None:
            attention_mask = (
                input_ids.ne(self.pad_token_id) if self.pad_token_id is not None else torch.ones_like(input_ids)
            )

        extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0

        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class RnaBertForMaskedLM(RnaBertPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaBertConfig, RnaBertForMaskedLM, RnaTokenizer
        >>> config = RnaBertConfig()
        >>> model = RnaBertForMaskedLM(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
    """

    def __init__(self, config: RnaBertConfig):
        super().__init__(config)
        self.rnabert = RnaBertModel(config, add_pooling_layer=False)
        self.lm_head = MaskedLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | torch.Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | MaskedLMOutput:
        outputs = self.rnabert(
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output = self.lm_head(outputs, labels)
        logits, loss = output.logits, output.loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaBertForPretraining(RnaBertPreTrainedModel):
    """
    Examples:
        >>> from multimolecule import RnaBertConfig, RnaBertForPretraining, RnaTokenizer
        >>> config = RnaBertConfig()
        >>> model = RnaBertForPretraining(config)
        >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
        >>> input = tokenizer("ACGUN", return_tensors="pt")
        >>> output = model(**input)
    """

    def __init__(self, config: RnaBertConfig):
        super().__init__(config)
        self.rnabert = RnaBertModel(config, add_pooling_layer=True)
        self.pretrain_head = RnaBertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | torch.Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        labels_ss: Tensor | None = None,
        next_sentence_label: Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | RnaBertForPretrainingOutput:
        outputs = self.rnabert(
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits, logits_ss, seq_relationship_score = self.pretrain_head(outputs)

        loss = None
        if any(x is not None for x in (labels, labels_ss, next_sentence_label)):
            loss_mlm = loss_ss = loss_nsp = 0
            if labels is not None:
                loss_mlm = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            if labels_ss is not None:
                loss_ss = F.cross_entropy(logits_ss.view(-1, self.config.ss_vocab_size), labels_ss.view(-1))
            if next_sentence_label is not None:
                loss_nsp = F.cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            loss = loss_mlm + loss_ss + loss_nsp

        if not return_dict:
            output = (logits, logits_ss) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RnaBertForPretrainingOutput(
            loss=loss,
            logits=logits,
            logits_ss=logits_ss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RnaBertEmbeddings(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        words_embeddings = self.word_embeddings(input_ids)

        # token type ids should have been unnecessary
        # added for consistency with original implementation
        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RnaBertEncoder(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RnaBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Tuple[Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for layer in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # type: ignore

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class RnaBertLayer(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.attention = RnaBertAttention(config)
        self.intermediate = RnaBertIntermediate(config)
        self.output = RnaBertOutput(config)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor, output_attentions: bool = False
    ) -> Tuple[Tensor, ...]:
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output, outputs = self_attention_outputs[0], self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class RnaBertAttention(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.selfattn = RnaBertSelfAttention(config)
        self.output = RnaBertSelfOutput(config)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor, output_attentions: bool = False
    ) -> Tuple[Tensor, ...]:
        self_outputs = self.selfattn(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RnaBertSelfAttention(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x: Tensor):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor, output_attentions: bool = False
    ) -> Tuple[Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = attention_scores.softmax(-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class RnaBertSelfOutput(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RnaBertIntermediate(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RnaBertOutput(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = RnaBertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RnaBertPooler(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RnaBertPreTrainingHeads(nn.Module):
    def __init__(self, config: RnaBertConfig):
        super().__init__()
        self.predictions = MaskedLMHead(config)
        vocab_size, config.vocab_size = config.vocab_size, config.ss_vocab_size
        self.predictions_ss = MaskedLMHead(config)
        config.vocab_size = vocab_size
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, outputs: ModelOutput | Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor]:
        sequence_output, pooled_output = outputs[:2]
        logits = self.predictions(sequence_output)
        logits_ss = self.predictions_ss(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return logits, logits_ss, seq_relationship_score


@dataclass
class RnaBertForPretrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None  # type: ignore[assignment]
    logits_ss: torch.FloatTensor = None  # type: ignore[assignment]
    hidden_states: Tuple[torch.FloatTensor, ...] | None = None
    attentions: Tuple[torch.FloatTensor, ...] | None = None


class RnaBertLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class RnaBertForSequenceClassification(RnaBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.rnabert = RnaBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #return_dict = None
        outputs = self.rnabert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
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
        # print(logits.shape,'33333333333333')
        # print(loss)
        # print(logits.shape)
        #print(len(outputs.hidden_states))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RnaBertForNucleotideLevel(RnaBertPreTrainedModel):
    # include Degradation and SpliceAI
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.rnabert = RnaBertModel(config)
       
        self.tokenizer = tokenizer
        if self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
            self.classifier_a = nn.Linear(config.hidden_size, config.num_labels)
            self.classifier_t = nn.Linear(config.hidden_size, config.num_labels)
            self.classifier_c = nn.Linear(config.hidden_size, config.num_labels)
            self.classifier_g = nn.Linear(config.hidden_size, config.num_labels)
            self.classifier_n = nn.Linear(config.hidden_size, config.num_labels)
            self.classifer_dict = {
                'A': self.classifier_a,
                'T': self.classifier_t,
                'C': self.classifier_c,
                'G': self.classifier_g,
                'N': self.classifier_n,
                }
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.rnabert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        final_input= outputs[0]

        ### init mappint tensor
        ori_length = weight_mask.shape[1]
        batch_size = final_input.shape[0]
        cur_length = int(final_input.shape[1])

        if self.config.token_type == 'single':
            assert attention_mask.shape==weight_mask.shape==post_token_length.shape
            mapping_final_input = final_input
        elif self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
            logits = torch.zeros((batch_size, ori_length, self.num_labels), dtype=final_input.dtype, device=final_input.device)
            nucleotide_indices = {nucleotide: (input_ids == self.tokenizer.encode(nucleotide, add_special_tokens=False)[0]).nonzero() for nucleotide in 'ATCGN'}
            mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
            for bz in range(batch_size):
                start_index = 0
                for i, length in enumerate(post_token_length[bz]): #astart from [cls]
                    mapping_final_input[bz,start_index:start_index + int(length.item()), :] = final_input[bz,i,:]
                    start_index += int(length.item())
            for nucleotide, indices in nucleotide_indices.items(): # indices:[bzid,seqid]
                #print(nucleotide, indices) 
                if indices.numel() > 0:  
                    bz_indices, pos_indices = indices.split(1, dim=1)
                    bz_indices = bz_indices.squeeze(-1) 
                    pos_indices = pos_indices.squeeze(-1)
                    nucleotide_logits = self.classifer_dict[nucleotide](mapping_final_input[bz_indices, pos_indices])
                    nucleotide_logits = nucleotide_logits.to(logits.dtype)
                    logits.index_put_((bz_indices, pos_indices), nucleotide_logits)
    
        elif 'mer' in self.config.token_type:
            kmer=int(self.config.token_type[0])
            mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
            mapping_final_input[:,0,:] = final_input[:,0,:] #[cls] token
            for bz in range(batch_size):
                value_length = torch.sum(attention_mask[bz,:]==1).item()
                for i in range(1,value_length-1): #exclude cls,sep token
                    mapping_final_input[bz,i:i+kmer,:] += final_input[bz,i]
                mapping_final_input[bz,value_length+kmer-1-1,:] = final_input[bz,value_length-1,:] #[sep] token
        #print(mapping_final_input.shape,weight_mask.shape)
        mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
        if 'mer' in self.config.token_type or self.config.token_type =='single': 
            logits = self.classifier(mapping_final_input)
        
        loss = None
        if labels is not None:
            logits = logits[:, 1:1+labels.size(1), :]
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MCRMSELoss()
                
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                #logits = logits[:, 1:1+labels.size(1), :]
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1).long())
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RnaBertForCRISPROffTarget(RnaBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.rnabert = RnaBertModel(config)


        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        sgrna_out = self.rnabert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=# position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )[1]
        target_out = self.rnabert(
            target_input_ids,
            attention_mask=target_attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=# position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )[1]
        final_input = torch.cat([sgrna_out,target_out],dim=-1)
        logits = self.classifier(final_input)
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
                loss_fct = nn.MSELoss()
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
        if not return_dict:
            output = (logits,) + sgrna_out[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
class RnaBertForStructuralimputation(RnaBertPreTrainedModel):
    # include Degradation and SpliceAI
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
        self.rnabert = RnaBertModel(config)
       
        self.tokenizer = tokenizer
        if self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
            self.down_mlp_a = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_t = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_c = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_g = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_n = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_dict = {
                'A': self.down_mlp_a,
                'T': self.down_mlp_t,
                'C': self.down_mlp_c,
                'G': self.down_mlp_g,
                'N': self.down_mlp_n,
                }
        else:
            self.down_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.embedding_struct = nn.Linear(1,config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        struct: Optional[torch.Tensor] = None,
        weight_mask: Optional[bool] = None,
        post_token_length: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.rnabert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        final_input= outputs[0]

        ### init mappint tensor
        ori_length = weight_mask.shape[1]
        batch_size = final_input.shape[0]
        cur_length = int(final_input.shape[1])

        if self.config.token_type == 'single':
            assert attention_mask.shape==weight_mask.shape==post_token_length.shape
            mapping_final_input = final_input
        elif self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
            inter_input = torch.zeros((batch_size, ori_length, self.config.hidden_size), dtype=final_input.dtype, device=final_input.device)
            nucleotide_indices = {nucleotide: (input_ids == self.tokenizer.encode(nucleotide, add_special_tokens=False)[0]).nonzero() for nucleotide in 'ATCGN'}
            mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
            for bz in range(batch_size):
                start_index = 0
                for i, length in enumerate(post_token_length[bz]): #astart from [cls]
                    mapping_final_input[bz,start_index:start_index + int(length.item()), :] = final_input[bz,i,:]
                    start_index += int(length.item())
            for nucleotide, indices in nucleotide_indices.items(): # indices:[bzid,seqid]
                #print(nucleotide, indices) 
                if indices.numel() > 0:  
                    bz_indices, pos_indices = indices.split(1, dim=1)
                    bz_indices = bz_indices.squeeze(-1) 
                    pos_indices = pos_indices.squeeze(-1)
                    nucleotide_logits = self.down_mlp_dict[nucleotide](mapping_final_input[bz_indices, pos_indices])
                    nucleotide_logits = nucleotide_logits.to(inter_input.dtype)
                    inter_input.index_put_((bz_indices, pos_indices), nucleotide_logits)
            #mapping_final_input = inter_input[:,1:-1,:]
        elif 'mer' in self.config.token_type:
            kmer=int(self.config.token_type[0])
            mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
            mapping_final_input[:,0,:] = final_input[:,0,:] #[cls] token
            for bz in range(batch_size):
                value_length = torch.sum(attention_mask[bz,:]==1).item()
                for i in range(1,value_length-1): #exclude cls,sep token
                    mapping_final_input[bz,i:i+kmer,:] += final_input[bz,i]
                mapping_final_input[bz,value_length+kmer-1-1,:] = final_input[bz,value_length-1,:] #[sep] token
        #print(mapping_final_input.shape,weight_mask.shape)
        mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
        
        if 'mer' in self.config.token_type or self.config.token_type =='single': 
            mapping_final_input = self.down_mlp(mapping_final_input)[:,1:-1,:] # exclude <cls> and <eos>
        elif self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
            mapping_final_input = mapping_final_input[:,1:-1,:]
        # print(labels.shape)
        # print(struct.shape,mapping_final_input.shape)
        struct_input = self.embedding_struct(struct.unsqueeze(-1))
        
        final_input = torch.cat([mapping_final_input,struct_input], dim=-1)

        logits = self.classifier(final_input)
        label_mask = struct== -1

    
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                print()
                if self.num_labels == 1:
                    loss = loss_fct(logits[label_mask].squeeze(), labels.squeeze())
                    # print(loss)
                else:
                    loss = loss_fct(logits[label_mask], labels)

        if not return_dict:
            output = (logits[label_mask],) + outputs[2:]
            return ((loss,) + output) if loss is not None else output       
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits[label_mask],
            hidden_states=None,
            attentions=None,
        )
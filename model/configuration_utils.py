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

from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass

from transformers.configuration_utils import PretrainedConfig as _PretrainedConfig


class PretrainedConfig(_PretrainedConfig):
    head: HeadConfig

    def __init__(
        self, pad_token_id=0, bos_token_id=1, eos_token_id=2, unk_token_id=3, mask_token_id=4, null_token_id=5, **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            mask_token_id=mask_token_id,
            null_token_id=null_token_id,
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        for k, v in output.items():
            if hasattr(v, "to_dict"):
                output[k] = v.to_dict()
            if is_dataclass(v):
                output[k] = asdict(v)
        return output


class BaseHeadConfig(OrderedDict):
    pass


@dataclass
class HeadConfig(BaseHeadConfig):
    r"""
    This is the configuration class to store the configuration of a prediction head. It is used to instantiate a
    prediction head according to the specified arguments, defining the head architecture.

    Configuration objects inherit from [`BaseHeadConfig`] and can be used to control the model outputs. Read the
    documentation from [`BaseHeadConfig`] for more information.


    Args:
        num_labels (`int`, *optional*):
            Number of labels to use in the last layer added to the model, typically for a classification task.
        problem_type (`str`, *optional*):
            Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
            `"single_label_classification"` or `"multi_label_classification"`.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        transform (`str`, *optional*, defaults to None):
            The transform operation applied to hidden states.
        transform_act (`str`, *optional*, defaults to "gelu"):
            The activation function of transform applied to hidden states.
        bias (`bool`, *optional*, defaults to True):
            Whether to apply bias to the final prediction layer.
        act (`str`, *optional*, defaults to None):
            The activation function of the final prediction output.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """

    num_labels: int = None  # type: ignore[assignment]
    problem_type: str = None  # type: ignore[assignment]
    hidden_size: int | None = None
    dropout: float = 0.0
    transform: str | None = None
    transform_act: str | None = "gelu"
    bias: bool = True
    act: str | None = None
    layer_norm_eps: float = 1e-12


@dataclass
class MaskedLMHeadConfig(BaseHeadConfig):
    r"""
    This is the configuration class to store the configuration of a prediction head. It is used to instantiate a
    prediction head according to the specified arguments, defining the head architecture.

    Configuration objects inherit from [`BaseHeadConfig`] and can be used to control the model outputs. Read the
    documentation from [`BaseHeadConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        transform (`str`, *optional*, defaults to "nonlinear"):
            The transform operation applied to hidden states.
        transform_act (`str`, *optional*, defaults to "gelu"):
            The activation function of transform applied to hidden states.
        bias (`bool`, *optional*, defaults to True):
            Whether to apply bias to the final prediction layer.
        act (`str`, *optional*, defaults to None):
            The activation function of the final prediction output.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """

    hidden_size: int | None = None
    dropout: float = 0.0
    transform: str | None = "nonlinear"
    transform_act: str | None = "gelu"
    bias: bool = True
    act: str | None = None
    layer_norm_eps: float = 1e-12

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

from typing import List, Sequence
from itertools import product
import torch
from torch import Tensor

torch.manual_seed(1013)

def generate_kmer_vocabulary(vocabulary: List[str], nmers: int = 1) -> List[str]:
    """
    Generates a kmer vocabulary given an original vocabulary and the size of kmers.

    Args:
        vocabulary (List[str]): The original vocabulary.
        nmers (int, defaults to 1): The size of the kmers to generate.

    Returns:
        vocabulary (List[str]): The kmer vocabulary.
    """

    if nmers <= 1:
        return vocabulary

    special_tokens, tokens = [], []
    for token in vocabulary:
        if token.startswith("<") or token.startswith("["):
            special_tokens.append(token)
        else:
            tokens.append(token)

    tokens = ["".join(kmer) for kmer in product(tokens, repeat=nmers)]

    return special_tokens + tokens

def get_vocab_list(nmers: int = 1, strameline: bool = False):
    vocab_list = STRAMELINE_VOCAB_LIST if strameline else VOCAB_LIST
    if nmers > 1:
        return generate_kmer_vocabulary(vocab_list, nmers)
    return vocab_list


def get_vocab_mapping():
    return VOCAB_MAPPING


def convert_word_embeddings(
    *old_embeddings: Tensor,
    old_vocab: List[str],
    new_vocab: List[str],
    mean: float = 0.0,
    std: float = 0.02,
    vocab_mapping: dict[str, str] | None = None,
) -> Sequence[Tensor]:
    if old_vocab == new_vocab:
        return old_embeddings
    if vocab_mapping is None:
        vocab_mapping = get_vocab_mapping()

    new_embeddings = []
    # Initialize the new embeddings
    for embeddings in old_embeddings:
        shape = embeddings.shape
        if shape[0] != len(old_vocab):
            raise ValueError("The first dimension of the embeddings must match the size of the vocabulary.")
        if embeddings.ndim == 1:  # Bias
            new_embeddings.append(torch.zeros(len(new_vocab)))
        else:
            new_embeddings.append(torch.normal(size=(len(new_vocab), *shape[1:]), mean=mean, std=std))

    # First Pass, copy the embeddings for the tokens that are in both vocabularies
    for old_index, old_token in enumerate(old_vocab):
        new_index = new_vocab.index(old_token)
        for new_embed, old_embed in zip(new_embeddings, old_embeddings):
            new_embed[new_index] = old_embed[old_index]

    # Second Pass, average the embeddings for the tokens that are in the new vocabulary but not in the old
    for token, tokens in vocab_mapping.items():
        if token not in new_vocab or token in old_vocab or len(tokens) == 1:
            continue
        index = new_vocab.index(token)
        indexes = [new_vocab.index(t) for t in tokens]
        for embed in new_embeddings:
            embed[index] = embed[indexes].mean(dim=0)

    return new_embeddings


def get_special_tokens_map():
    return SPECIAL_TOKENS_MAP


def get_tokenizer_config():
    config = TOKENIZER_CONFIG
    config.setdefault("added_tokens_decoder", {})
    for i, v in enumerate(SPECIAL_TOKENS_MAP.values()):
        config["added_tokens_decoder"][str(i)] = v
    return config


STRAMELINE_VOCAB_LIST = [
    "<pad>",
    "<cls>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<null>",
    "A",
    "C",
    "G",
    "T",
    "N",
]


VOCAB_LIST = [
    "<pad>",
    "<cls>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<null>",
    "A",
    "C",
    "G",
    "T",
    "N",
    "X",
    "V",
    "H",
    "D",
    "B",
    "M",
    "R",
    "W",
    "S",
    "Y",
    "K",
    ".",
    "*",
    "-",
]

VOCAB_MAPPING = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "N": "N",
    "X": "ACGU",
    "V": "ACG",
    "H": "ACU",
    "D": "AGU",
    "B": "CGU",
    "M": "AC",
    "R": "AG",
    "W": "AU",
    "S": "CG",
    "Y": "CU",
    "K": "GU",
    ".": ".",
    "*": "*",
    "-": "-",
}

SPECIAL_TOKENS_MAP = {
    "pad_token": {
        "content": "<pad>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "cls_token": {
        "content": "<cls>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "eos_token": {
        "content": "<eos>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "unk_token": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
    "mask_token": {
        "content": "<mask>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    },
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "RnaTokenizer",
    "clean_up_tokenization_spaces": True,
}

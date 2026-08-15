"""Trusted local adapter loading and bounded execution configuration."""

from __future__ import annotations

import importlib
from typing import cast

import torch

from modelpact.adapters.base import ModelAdapter
from modelpact.adapters.huggingface import HuggingFaceCausalLMAdapter
from modelpact.adapters.tiny_lm import TinyModelAdapter


class AdapterLoadError(ValueError):
    pass


def load_trusted_adapter(specification: str) -> ModelAdapter:
    """Load a built-in or explicitly trusted ``module:attribute`` adapter.

    Importing a custom module executes local Python. This function is never used
    on an adapter reference embedded in an untrusted patch bundle.
    """

    if specification == "tiny":
        return TinyModelAdapter()
    if specification in {"huggingface", "hf"}:
        return HuggingFaceCausalLMAdapter()
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise AdapterLoadError("adapter must be 'tiny', 'huggingface', or trusted 'module:attribute'")
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute_name)
        candidate = value() if isinstance(value, type) else value
    except (ImportError, AttributeError, TypeError) as error:
        raise AdapterLoadError(f"could not load trusted adapter {specification!r}") from error
    if not isinstance(candidate, ModelAdapter):
        raise AdapterLoadError(f"object does not implement ModelAdapter: {specification!r}")
    return cast(ModelAdapter, candidate)


def parse_dtype(value: str) -> torch.dtype:
    types = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return types[value.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {value!r}; choose {', '.join(types)}") from error

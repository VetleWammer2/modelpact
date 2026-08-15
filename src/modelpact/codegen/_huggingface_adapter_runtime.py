#!/usr/bin/env python3
"""Reviewed source fragment embedded in standalone Hugging Face verifiers.

This module is package-owned code, never copied into or imported from an
untrusted patch bundle. Generated verifiers expose it only through the explicit
``--adapter-kind huggingface`` selection. The implementation accepts local,
unsharded SafeTensors checkpoints and disables every Transformers network and
remote-code path.
"""

from __future__ import annotations

# MODELPACT_BUILTIN_HUGGINGFACE_ADAPTER_BEGIN
import json as _hf_json_module
import math as _hf_math
import os as _hf_os
from collections.abc import Sequence as _HFSequence
from dataclasses import dataclass as _hf_dataclass
from pathlib import Path as _HFPath
from typing import Protocol as _HFProtocol
from typing import cast as _hf_cast

import torch as _hf_torch
from safetensors import safe_open as _hf_safe_open
from torch import Tensor as _HFTensor
from torch import nn as _hf_nn

_HF_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_HF_MAX_CHECKPOINT_BYTES = 16 * 1024**3
_HF_MAX_FILES = 10_000
_HF_MAX_TENSORS = 100_000
_HF_MAX_GENERATED_TOKENS = 4096
_HF_UNSAFE_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".h5", ".msgpack", ".pickle", ".pkl", ".pt", ".pth"}
)


def _hf_unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Hugging Face configuration key: {key!r}")
        result[key] = value
    return result


def _hf_reject_constant(value: str) -> None:
    raise ValueError(f"non-finite Hugging Face configuration value: {value}")


def _hf_validate_depth(value: object, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("Hugging Face configuration exceeds the nesting limit")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Hugging Face configuration keys must be strings")
            _hf_validate_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _hf_validate_depth(item, depth + 1)


def _hf_read_configuration(path: _HFPath) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("local Hugging Face checkpoint requires a regular config.json")
    if path.stat().st_size > _HF_MAX_CONFIG_BYTES:
        raise ValueError("Hugging Face configuration exceeds the size limit")
    try:
        value = _hf_json_module.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_hf_unique_object,
            parse_constant=_hf_reject_constant,
        )
    except (OSError, UnicodeDecodeError, _hf_json_module.JSONDecodeError) as error:
        raise ValueError("malformed Hugging Face configuration") from error
    if not isinstance(value, dict):
        raise ValueError("Hugging Face configuration must be an object")
    _hf_validate_depth(value)
    if not isinstance(value.get("model_type"), str):
        raise ValueError("Hugging Face configuration has no model_type")
    return value


def _hf_checkpoint_root(checkpoint: str) -> _HFPath:
    root = _HFPath(checkpoint)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Hugging Face checkpoint must be a local regular directory")
    _hf_read_configuration(root / "config.json")
    entries = list(root.rglob("*"))
    if len(entries) > _HF_MAX_FILES:
        raise ValueError("Hugging Face checkpoint contains too many files")
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("Hugging Face checkpoint may not contain symlinks")
        if entry.is_file() and entry.suffix.lower() in _HF_UNSAFE_WEIGHT_SUFFIXES:
            raise ValueError("Hugging Face checkpoint contains a non-SafeTensors weight file")
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("standalone Hugging Face verification does not support sharded weights")
    tensors = sorted(root.glob("*.safetensors"))
    if len(tensors) != 1 or tensors[0].name != "model.safetensors":
        raise ValueError("Hugging Face checkpoint requires exactly one model.safetensors file")
    tensor_path = tensors[0]
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("Hugging Face SafeTensors checkpoint must be a regular file")
    if tensor_path.stat().st_size > _HF_MAX_CHECKPOINT_BYTES:
        raise ValueError("Hugging Face SafeTensors checkpoint exceeds the size limit")
    try:
        # SafeTensors does not currently publish callable typing for safe_open.
        with _hf_safe_open(  # type: ignore[no-untyped-call]
            tensor_path, framework="pt", device="cpu"
        ) as handle:
            keys = handle.keys()
            if not keys or len(keys) > _HF_MAX_TENSORS:
                raise ValueError("Hugging Face checkpoint has an invalid tensor count")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("invalid Hugging Face SafeTensors checkpoint") from error
    return root.resolve(strict=True)


class _HFGenerationPolicyLike(_HFProtocol):
    mode: str
    max_new_tokens: int
    seed: int
    temperature: float
    top_k: int | None
    top_p: float
    stop_on_eos: bool


@_hf_dataclass(frozen=True, slots=True)
class HuggingFaceModelBatch:
    input_ids: _HFTensor
    attention_mask: _HFTensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input IDs and attention mask must be equally shaped matrices")
        if self.input_ids.dtype != _hf_torch.long or self.attention_mask.dtype != _hf_torch.bool:
            raise ValueError("invalid Hugging Face batch dtypes")

    def to(self, device: _hf_torch.device | str) -> HuggingFaceModelBatch:
        return HuggingFaceModelBatch(self.input_ids.to(device), self.attention_mask.to(device))


@_hf_dataclass(frozen=True, slots=True)
class HuggingFaceGeneratedSample:
    token_ids: tuple[int, ...]
    text: str
    finished: bool


class StandaloneHuggingFaceTokenizer:
    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer

    @property
    def pad_token_id(self) -> int:
        value = getattr(self._tokenizer, "pad_token_id", None)
        if value is None:
            value = getattr(self._tokenizer, "eos_token_id", None)
        if value is None:
            raise ValueError("local tokenizer has neither a pad token nor an EOS fallback")
        return int(value)

    @property
    def bos_token_id(self) -> int:
        value = getattr(self._tokenizer, "bos_token_id", None)
        if value is None:
            raise ValueError("local tokenizer has no BOS token")
        return int(value)

    @property
    def eos_token_id(self) -> int:
        value = getattr(self._tokenizer, "eos_token_id", None)
        if value is None:
            raise ValueError("local tokenizer has no EOS token")
        return int(value)

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        encode = getattr(self._tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("local tokenizer has no encode method")
        tokens = [int(token) for token in encode(text, add_special_tokens=False)]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: _HFSequence[int], *, skip_special_tokens: bool = True) -> str:
        decode = getattr(self._tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("local tokenizer has no decode method")
        return str(decode(list(token_ids), skip_special_tokens=skip_special_tokens))

    def batch(self, texts: _HFSequence[str], *, add_bos: bool = True) -> HuggingFaceModelBatch:
        if not texts:
            raise ValueError("cannot encode an empty batch")
        encoded = [self.encode(text, add_bos=add_bos) for text in texts]
        width = max(map(len, encoded))
        ids = _hf_torch.full((len(encoded), width), self.pad_token_id, dtype=_hf_torch.long)
        mask = _hf_torch.zeros_like(ids, dtype=_hf_torch.bool)
        for row, tokens in enumerate(encoded):
            start = width - len(tokens)
            ids[row, start:] = _hf_torch.tensor(tokens, dtype=_hf_torch.long)
            mask[row, start:] = True
        return HuggingFaceModelBatch(ids, mask)


class StandaloneHuggingFaceModelAdapter:
    adapter_id = "modelpact.huggingface_causal_lm.local.v1"

    def __init__(self) -> None:
        self._tokenizer_adapter: StandaloneHuggingFaceTokenizer | None = None

    def load(
        self,
        checkpoint: str,
        *,
        device: _hf_torch.device | str = "cpu",
        dtype: _hf_torch.dtype = _hf_torch.float32,
    ) -> _hf_nn.Module:
        root = _hf_checkpoint_root(checkpoint)
        # Enforce offline behavior in this process even when the caller did not
        # configure the conventional Hugging Face environment variables.
        _hf_os.environ["HF_HUB_OFFLINE"] = "1"
        _hf_os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _hf_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "standalone Hugging Face verification requires Transformers and its "
                f"local dependencies: {error}"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        if not isinstance(loaded, _hf_nn.Module):
            raise TypeError("Transformers returned a non-module causal LM")
        model_config = getattr(loaded, "config", None)
        if getattr(model_config, "is_encoder_decoder", False) is True:
            raise ValueError("checkpoint is not a decoder-only causal language model")
        if model_config is not None:
            model_config.use_cache = False
        self._tokenizer_adapter = StandaloneHuggingFaceTokenizer(tokenizer)
        model = _hf_cast(_hf_nn.Module, loaded)
        model.eval()
        return model.to(device)

    def tokenizer(self) -> StandaloneHuggingFaceTokenizer:
        if self._tokenizer_adapter is None:
            raise RuntimeError("load a checkpoint before requesting its tokenizer")
        return self._tokenizer_adapter

    def prepare(self, model: _hf_nn.Module) -> None:
        model.eval()
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

    def forward_logits(self, model: _hf_nn.Module, batch: HuggingFaceModelBatch) -> _HFTensor:
        device = next(model.parameters()).device
        moved = batch.to(device)
        output = model(input_ids=moved.input_ids, attention_mask=moved.attention_mask)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, _HFTensor):
            raise TypeError("causal-LM adapter expected tensor logits")
        return logits

    def generate(
        self,
        model: _hf_nn.Module,
        batch: HuggingFaceModelBatch,
        policy: _HFGenerationPolicyLike,
    ) -> list[HuggingFaceGeneratedSample]:
        if policy.mode not in {"greedy", "sample"}:
            raise ValueError("unsupported generation mode")
        if not 1 <= policy.max_new_tokens <= _HF_MAX_GENERATED_TOKENS:
            raise ValueError("generation token budget is outside limits")
        if not _hf_math.isfinite(policy.temperature) or policy.temperature <= 0:
            raise ValueError("generation temperature must be finite and positive")
        if not _hf_math.isfinite(policy.top_p) or not 0 < policy.top_p <= 1:
            raise ValueError("generation top_p must be in (0, 1]")
        if policy.top_k is not None and policy.top_k < 1:
            raise ValueError("generation top_k must be positive")
        tokenizer = self.tokenizer()
        device = next(model.parameters()).device
        moved = batch.to(device)
        prior_mode = model.training
        model.eval()
        try:
            fork_devices = (
                [device.index if device.index is not None else _hf_torch.cuda.current_device()]
                if device.type == "cuda"
                else []
            )
            with _hf_torch.no_grad(), _hf_torch.random.fork_rng(devices=fork_devices):
                _hf_torch.manual_seed(policy.seed)
                if device.type == "cuda":
                    _hf_torch.cuda.manual_seed_all(policy.seed)
                generate = getattr(model, "generate", None)
                if not callable(generate):
                    raise TypeError("causal-LM module has no generate method")
                arguments: dict[str, object] = {
                    "input_ids": moved.input_ids,
                    "attention_mask": moved.attention_mask,
                    "do_sample": policy.mode == "sample",
                    "max_new_tokens": policy.max_new_tokens,
                    "eos_token_id": (tokenizer.eos_token_id if policy.stop_on_eos else None),
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": False,
                }
                if policy.mode == "sample":
                    arguments["temperature"] = policy.temperature
                    arguments["top_p"] = policy.top_p
                    if policy.top_k is not None:
                        arguments["top_k"] = policy.top_k
                output = generate(**arguments)
        finally:
            model.train(prior_mode)
        if not isinstance(output, _HFTensor):
            raise TypeError("causal-LM generate returned a non-tensor result")
        prompt_width = moved.input_ids.shape[1]
        samples: list[HuggingFaceGeneratedSample] = []
        for row in range(output.shape[0]):
            token_ids = tuple(int(token) for token in output[row, prompt_width:].tolist())
            finished = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            samples.append(
                HuggingFaceGeneratedSample(token_ids, tokenizer.decode(token_ids), finished)
            )
        return samples


# MODELPACT_BUILTIN_HUGGINGFACE_ADAPTER_END

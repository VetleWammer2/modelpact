"""Safe, local-only Hugging Face causal-LM adapter.

Transformers is optional.  This adapter never downloads files and never enables
remote model code.  The checkpoint path and adapter module are trusted local
inputs; Hugging Face checkpoint metadata remains untrusted data.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import Tensor, nn

from modelpact.adapters.base import (
    ActivationPoint,
    GeneratedSample,
    GenerationPolicy,
    ModelBatch,
    PatchableModule,
)
from modelpact.checkpoints.store import checkpoint_files
from modelpact.models.schema import ModelStateSchema, inspect_state_schema
from modelpact.util.canonical_json import strict_json_loads

_MAX_CONFIG_BYTES = 16 * 1024**2
_MAX_CHECKPOINT_FILES = 100_000
_MAX_CHECKPOINT_SHARDS = 10_000
_MAX_CHECKPOINT_TENSORS = 100_000
_MAX_SHARD_BYTES = 16 * 1024**3
_UNSAFE_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".h5", ".msgpack", ".pickle", ".pkl", ".pt", ".pth"}
)


def _checkpoint_entries(root: Path) -> tuple[Path, ...]:
    resolved_root = root.resolve()
    pending = [root]
    entries: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as error:
            raise ValueError(
                f"cannot inspect local Hugging Face checkpoint: {directory}"
            ) from error
        for child in children:
            entries.append(child)
            if len(entries) > _MAX_CHECKPOINT_FILES:
                raise ValueError("Hugging Face checkpoint contains too many files")
            if child.is_symlink():
                raise ValueError("Hugging Face checkpoint may not contain symlinks")
            resolved = child.resolve()
            if resolved_root != resolved and resolved_root not in resolved.parents:
                raise ValueError("Hugging Face checkpoint entry escapes its directory")
            if child.is_dir():
                pending.append(child)
    return tuple(entries)


def _validate_local_checkpoint(checkpoint: str) -> Path:
    root = Path(checkpoint)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Hugging Face checkpoint must be a local regular directory")
    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("local Hugging Face checkpoint requires a regular config.json")
    if config_path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("Hugging Face configuration exceeds the size limit")
    configuration = strict_json_loads(config_path.read_bytes(), max_depth=32)
    if not isinstance(configuration, dict):
        raise ValueError("Hugging Face configuration must be an object")

    entries = _checkpoint_entries(root)
    for entry in entries:
        if not entry.is_file():
            continue
        lowered = entry.name.lower()
        if entry.suffix.lower() in _UNSAFE_WEIGHT_SUFFIXES or lowered.endswith(
            tuple(f"{suffix}.index.json" for suffix in _UNSAFE_WEIGHT_SUFFIXES)
        ):
            raise ValueError("Hugging Face checkpoint contains a non-SafeTensors weight file")

    shards = checkpoint_files(root)
    if len(shards) > _MAX_CHECKPOINT_SHARDS:
        raise ValueError("Hugging Face checkpoint contains too many shards")
    index = root / "model.safetensors.index.json"
    if not index.exists() and (
        len(shards) != 1 or shards[0].resolve() != (root / "model.safetensors").resolve()
    ):
        raise ValueError("unsharded Hugging Face checkpoint requires exactly one model.safetensors")
    tensor_count = 0
    for shard in shards:
        if shard.suffix.lower() != ".safetensors" or not shard.is_file():
            raise ValueError("checkpoint index must reference regular SafeTensors shards")
        if shard.stat().st_size > _MAX_SHARD_BYTES:
            raise ValueError("Hugging Face checkpoint shard exceeds the size limit")
        try:
            # SafeTensors does not currently publish callable typing for safe_open.
            with safe_open(  # type: ignore[no-untyped-call]
                shard, framework="pt", device="cpu"
            ) as handle:
                tensor_count += len(handle.keys())
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"invalid SafeTensors checkpoint shard: {shard.name}") from error
        if tensor_count > _MAX_CHECKPOINT_TENSORS:
            raise ValueError("Hugging Face checkpoint contains too many tensors")
    if tensor_count == 0:
        raise ValueError("Hugging Face checkpoint contains no tensors")
    return root


class HuggingFaceTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def pad_token_id(self) -> int:
        value = self._tokenizer.pad_token_id
        if value is None:
            value = self._tokenizer.eos_token_id
        if value is None:
            raise ValueError("local tokenizer has neither a pad token nor an EOS fallback")
        return int(value)

    @property
    def bos_token_id(self) -> int:
        value = self._tokenizer.bos_token_id
        if value is None:
            raise ValueError("local tokenizer has no BOS token")
        return int(value)

    @property
    def eos_token_id(self) -> int:
        value = self._tokenizer.eos_token_id
        if value is None:
            raise ValueError("local tokenizer has no EOS token")
        return int(value)

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        tokens = list(self._tokenizer.encode(text, add_special_tokens=False))
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return [int(token) for token in tokens]

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens))

    def batch(self, texts: Sequence[str], *, add_bos: bool = True) -> ModelBatch:
        if not texts:
            raise ValueError("cannot encode an empty batch")
        encoded = [self.encode(text, add_bos=add_bos) for text in texts]
        width = max(map(len, encoded))
        ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, tokens in enumerate(encoded):
            # Decoder-only generation is least surprising with left padding.
            start = width - len(tokens)
            ids[row, start:] = torch.tensor(tokens)
            mask[row, start:] = True
        return ModelBatch(ids, mask)


class HuggingFaceCausalLMAdapter:
    adapter_id = "modelpact.huggingface_causal_lm.local.v1"
    supports_sampling_controls = True

    def __init__(self) -> None:
        self._tokenizer_adapter: HuggingFaceTokenizerAdapter | None = None

    @staticmethod
    def _local_directory(checkpoint: str) -> Path:
        return _validate_local_checkpoint(checkpoint)

    def load(
        self,
        checkpoint: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> nn.Module:
        root = self._local_directory(checkpoint)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face support requires the optional 'huggingface' dependency"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        if not isinstance(loaded, nn.Module):
            raise TypeError("Transformers returned a non-module causal LM")
        model = cast(nn.Module, loaded)
        model_config = getattr(model, "config", None)
        # Decoder-only configs such as GPT-2 and GPT-NeoX commonly leave
        # ``is_decoder`` false; that flag describes cross-attention behavior,
        # not the AutoModelForCausalLM architecture class.
        if getattr(model_config, "is_encoder_decoder", False) is True:
            raise ValueError("checkpoint is not a decoder-only causal language model")
        self._tokenizer_adapter = HuggingFaceTokenizerAdapter(tokenizer)
        return model.to(device)

    def tokenizer(self) -> HuggingFaceTokenizerAdapter:
        if self._tokenizer_adapter is None:
            raise RuntimeError("load a checkpoint before requesting its tokenizer")
        return self._tokenizer_adapter

    def prepare(self, model: nn.Module) -> None:
        model.eval()
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

    def forward_logits(self, model: nn.Module, batch: ModelBatch) -> Tensor:
        device = next(model.parameters()).device
        moved = batch.to(device)
        output = model(input_ids=moved.input_ids, attention_mask=moved.attention_mask)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise TypeError("causal-LM adapter expected tensor logits")
        return logits

    def generate(
        self, model: nn.Module, batch: ModelBatch, policy: GenerationPolicy
    ) -> list[GeneratedSample]:
        tokenizer = self.tokenizer()
        device = next(model.parameters()).device
        moved = batch.to(device)
        prior_mode = model.training
        model.eval()
        try:
            fork_devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda"
                else []
            )
            with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(policy.seed)
                generate = getattr(model, "generate", None)
                if not callable(generate):
                    raise TypeError("causal-LM module has no generate method")
                generation_arguments: dict[str, object] = {
                    "input_ids": moved.input_ids,
                    "attention_mask": moved.attention_mask,
                    "do_sample": policy.mode == "sample",
                    "max_new_tokens": policy.max_new_tokens,
                    "eos_token_id": tokenizer.eos_token_id if policy.stop_on_eos else None,
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": False,
                }
                if policy.mode == "sample":
                    generation_arguments["temperature"] = policy.temperature
                    generation_arguments["top_p"] = policy.top_p
                    if policy.top_k is not None:
                        generation_arguments["top_k"] = policy.top_k
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(policy.seed)
                output = generate(**generation_arguments)
        finally:
            model.train(prior_mode)
        if not isinstance(output, Tensor):
            raise TypeError("causal-LM generate returned a non-tensor result")
        samples = []
        for row in range(output.shape[0]):
            prompt_length = moved.input_ids.shape[1]
            token_ids = tuple(int(token) for token in output[row, prompt_length:].tolist())
            finished = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            samples.append(GeneratedSample(token_ids, tokenizer.decode(token_ids), finished))
        return samples

    def patchable_modules(self, model: nn.Module) -> Iterable[PatchableModule]:
        for path, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names = ("weight",) + (("bias",) if module.bias is not None else ())
                yield PatchableModule(path, module, names, "linear")
            elif isinstance(module, nn.Embedding):
                yield PatchableModule(path, module, ("weight",), "embedding")
            elif isinstance(module, nn.LayerNorm) or "rmsnorm" in type(module).__name__.lower():
                names = tuple(name for name, _ in module.named_parameters(recurse=False))
                yield PatchableModule(path, module, names, "norm")

    def activation_points(self, model: nn.Module) -> Iterable[ActivationPoint]:
        # Architecture-neutral default: transformer block-like modules are
        # explicit candidates; custom adapters can provide stronger semantics.
        for path, module in model.named_modules():
            lowered = type(module).__name__.lower()
            if "decoderlayer" in lowered or lowered.endswith("block"):
                yield ActivationPoint(path, module, "residual_stream_candidate")

    def state_schema(self, model: nn.Module) -> ModelStateSchema:
        return inspect_state_schema(model)

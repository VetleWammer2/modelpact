"""Safe, local-only Hugging Face causal-LM adapter.

Transformers is optional.  This adapter never downloads files and never enables
remote model code.  The checkpoint path and adapter module are trusted local
inputs; Hugging Face checkpoint metadata remains untrusted data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from modelpact.adapters.base import (
    ActivationPoint,
    GeneratedSample,
    GenerationPolicy,
    ModelBatch,
    PatchableModule,
)
from modelpact.models.schema import ModelStateSchema, inspect_state_schema


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

    def __init__(self) -> None:
        self._tokenizer_adapter: HuggingFaceTokenizerAdapter | None = None

    @staticmethod
    def _local_directory(checkpoint: str) -> Path:
        root = Path(checkpoint)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Hugging Face checkpoint must be a local regular directory")
        if not (root / "config.json").is_file():
            raise ValueError("local Hugging Face checkpoint has no config.json")
        return root

    def load(
        self,
        checkpoint: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> nn.Module:
        root = self._local_directory(checkpoint)
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
        if getattr(model_config, "is_decoder", True) is False:
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
                }
                if policy.mode == "sample":
                    generation_arguments["temperature"] = policy.temperature
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

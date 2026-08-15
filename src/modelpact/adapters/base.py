"""Narrow trusted-local-code model adapter protocol.

Model adapters execute arbitrary local Python and are inside ModelPact's trust
boundary. Patch bundles, contracts, manifests, and certificates are data and
must never be treated as adapters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from modelpact.models.schema import ModelStateSchema


@dataclass(frozen=True, slots=True)
class ModelBatch:
    input_ids: Tensor
    attention_mask: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input_ids and attention_mask must be equally shaped rank-2 tensors")
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long")
        if self.attention_mask.dtype not in {torch.bool, torch.long}:
            raise ValueError("attention_mask must be boolean or long")

    def to(self, device: torch.device | str) -> ModelBatch:
        return ModelBatch(self.input_ids.to(device), self.attention_mask.to(device))


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    mode: str = "greedy"
    max_new_tokens: int = 32
    seed: int = 0
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    stop_on_eos: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"greedy", "sample"}:
            raise ValueError(f"unsupported generation mode: {self.mode}")
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        if self.temperature <= 0 or not torch.isfinite(torch.tensor(self.temperature)):
            raise ValueError("temperature must be finite and positive")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not 1 <= self.top_k <= 10_000_000
        ):
            raise ValueError("top_k must be in [1, 10000000]")
        if (
            isinstance(self.top_p, bool)
            or not 0 < self.top_p <= 1
            or not torch.isfinite(torch.tensor(self.top_p))
        ):
            raise ValueError("top_p must be finite and in (0, 1]")


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    token_ids: tuple[int, ...]
    text: str
    finished: bool


@dataclass(frozen=True, slots=True)
class PatchableModule:
    path: str
    module: nn.Module
    parameter_names: tuple[str, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class ActivationPoint:
    path: str
    module: nn.Module
    semantic: str


@runtime_checkable
class TokenizerAdapter(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_token_id(self) -> int: ...

    @property
    def bos_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...

    def batch(self, texts: Sequence[str], *, add_bos: bool = True) -> ModelBatch: ...


@runtime_checkable
class ModelAdapter(Protocol):
    adapter_id: str

    def load(
        self,
        checkpoint: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> nn.Module: ...

    def tokenizer(self) -> TokenizerAdapter: ...

    def prepare(self, model: nn.Module) -> None: ...

    def forward_logits(self, model: nn.Module, batch: ModelBatch) -> Tensor: ...

    def generate(
        self, model: nn.Module, batch: ModelBatch, policy: GenerationPolicy
    ) -> list[GeneratedSample]: ...

    def patchable_modules(self, model: nn.Module) -> Iterable[PatchableModule]: ...

    def activation_points(self, model: nn.Module) -> Iterable[ActivationPoint]: ...

    def state_schema(self, model: nn.Module) -> ModelStateSchema: ...

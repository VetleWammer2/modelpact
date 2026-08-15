"""Model Manifest v1 and its stable signature."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from modelpact.models.fingerprint import (
    canonical_config,
    chat_template_fingerprint,
    checkpoint_tensor_fingerprint,
    generation_config_fingerprint,
    tokenizer_fingerprint,
)
from modelpact.models.schema import ModelStateSchema, inspect_state_schema
from modelpact.util.hashing import hash_canonical


def _hash_text(value: object) -> str:
    return hash_canonical(value)


@dataclass(frozen=True, slots=True)
class ModelSignature:
    schema_version: int
    adapter_id: str
    architecture_hash: str
    state_schema_hash: str
    checkpoint_hash: str
    tokenizer_hash: str
    chat_template_hash: str
    generation_config_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError(f"unsupported model signature version: {self.schema_version}")
        if not self.adapter_id:
            raise ValueError("adapter_id is required")
        for name in (
            "architecture_hash",
            "state_schema_hash",
            "checkpoint_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "generation_config_hash",
        ):
            if not str(getattr(self, name)).startswith("sha256:"):
                raise ValueError(f"{name} must be a tagged SHA-256 digest")

    @property
    def signature_hash(self) -> str:
        return hash_canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "architecture_hash": self.architecture_hash,
            "chat_template_hash": self.chat_template_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "generation_config_hash": self.generation_config_hash,
            "schema_version": self.schema_version,
            "state_schema_hash": self.state_schema_hash,
            "tokenizer_hash": self.tokenizer_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelSignature:
        required = {
            "schema_version": int,
            "adapter_id": str,
            "architecture_hash": str,
            "state_schema_hash": str,
            "checkpoint_hash": str,
            "tokenizer_hash": str,
            "chat_template_hash": str,
            "generation_config_hash": str,
        }
        parsed: dict[str, Any] = {}
        for key, expected in required.items():
            item = value.get(key)
            if not isinstance(item, expected):
                raise ValueError(f"invalid model signature field: {key}")
            parsed[key] = item
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    signature: ModelSignature
    state_schema: ModelStateSchema
    checkpoint_tensor_hashes: Mapping[str, str]
    parameter_count: int
    patchable_parameter_count: int
    dtype_distribution: Mapping[str, int]
    supported_runtime_modes: tuple[str, ...] = ("runtime_mount", "materialize")
    unsupported_state: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError(f"unsupported model manifest version: {self.schema_version}")
        if self.signature.state_schema_hash != self.state_schema.schema_hash:
            raise ValueError("signature/state schema hash mismatch")
        if (
            self.parameter_count < 0
            or not 0 <= self.patchable_parameter_count <= self.parameter_count
        ):
            raise ValueError("invalid parameter counts")

    @property
    def manifest_hash(self) -> str:
        return hash_canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_tensor_hashes": dict(sorted(self.checkpoint_tensor_hashes.items())),
            "dtype_distribution": dict(sorted(self.dtype_distribution.items())),
            "metadata": dict(self.metadata),
            "parameter_count": self.parameter_count,
            "patchable_parameter_count": self.patchable_parameter_count,
            "schema_version": self.schema_version,
            "signature": self.signature.to_dict(),
            "state_schema": self.state_schema.to_dict(),
            "supported_runtime_modes": list(self.supported_runtime_modes),
            "unsupported_state": list(self.unsupported_state),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelManifest:
        signature = value.get("signature")
        state_schema = value.get("state_schema")
        tensor_hashes = value.get("checkpoint_tensor_hashes")
        dtype_distribution = value.get("dtype_distribution")
        runtime = value.get("supported_runtime_modes", [])
        unsupported = value.get("unsupported_state", [])
        metadata = value.get("metadata", {})
        if (
            value.get("schema_version") != 1
            or not isinstance(signature, Mapping)
            or not isinstance(state_schema, Mapping)
            or not isinstance(tensor_hashes, Mapping)
            or not all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in tensor_hashes.items()
            )
            or not isinstance(dtype_distribution, Mapping)
            or not all(
                isinstance(key, str) and isinstance(item, int)
                for key, item in dtype_distribution.items()
            )
            or not isinstance(runtime, list)
            or not all(isinstance(item, str) for item in runtime)
            or not isinstance(unsupported, list)
            or not all(isinstance(item, str) for item in unsupported)
            or not isinstance(metadata, Mapping)
        ):
            raise ValueError("malformed model manifest")
        parameter_count = value.get("parameter_count")
        patchable_count = value.get("patchable_parameter_count")
        if not isinstance(parameter_count, int) or not isinstance(patchable_count, int):
            raise ValueError("invalid model manifest parameter counts")
        return cls(
            schema_version=1,
            signature=ModelSignature.from_dict(signature),
            state_schema=ModelStateSchema.from_dict(state_schema),
            checkpoint_tensor_hashes=dict(tensor_hashes),
            parameter_count=parameter_count,
            patchable_parameter_count=patchable_count,
            dtype_distribution=dict(dtype_distribution),
            supported_runtime_modes=tuple(runtime),
            unsupported_state=tuple(unsupported),
            metadata=dict(metadata),
        )


def build_model_manifest(
    model: nn.Module,
    *,
    checkpoint: str | Path,
    adapter_id: str,
    architecture_config: Mapping[str, object] | None = None,
) -> ModelManifest:
    """Build an identity manifest from a loaded model and its local files."""

    schema = inspect_state_schema(model)
    checkpoint_hash, tensor_hashes = checkpoint_tensor_fingerprint(checkpoint)
    config = (
        architecture_config if architecture_config is not None else canonical_config(checkpoint)
    )
    architecture_hash = _hash_text(
        {
            "adapter_id": adapter_id,
            "configuration": config,
            "modules": [m.to_dict() for m in schema.modules],
        }
    )
    signature = ModelSignature(
        schema_version=1,
        adapter_id=adapter_id,
        architecture_hash=architecture_hash,
        state_schema_hash=schema.schema_hash,
        checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_fingerprint(checkpoint),
        chat_template_hash=chat_template_fingerprint(checkpoint),
        generation_config_hash=generation_config_fingerprint(checkpoint),
    )
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    patchable_names = {item.name for item in schema.tensors if item.patchable}
    by_name = dict(model.named_parameters(remove_duplicate=False))
    # Count aliases only once by object identity.
    seen: set[int] = set()
    patchable_count = 0
    for name in sorted(patchable_names):
        parameter = by_name[name]
        if id(parameter) not in seen:
            seen.add(id(parameter))
            patchable_count += parameter.numel()
    dtype_counts = Counter(str(parameter.dtype).removeprefix("torch.") for parameter in parameters)
    return ModelManifest(
        schema_version=1,
        signature=signature,
        state_schema=schema,
        checkpoint_tensor_hashes=tensor_hashes,
        parameter_count=parameter_count,
        patchable_parameter_count=patchable_count,
        dtype_distribution=dict(sorted(dtype_counts.items())),
    )

"""SafeTensors checkpoint readers and deterministic writers."""

from modelpact.checkpoints.safetensors import (
    load_safetensors,
    save_safetensors_atomic,
    tensor_content_hash,
)


def materialize_checkpoint(*args: object, **kwargs: object) -> dict[str, object]:
    """Lazily import the writer to avoid coupling schema inspection to patch IR."""

    from modelpact.checkpoints.writer import materialize_checkpoint as implementation

    return implementation(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "load_safetensors",
    "materialize_checkpoint",
    "save_safetensors_atomic",
    "tensor_content_hash",
]

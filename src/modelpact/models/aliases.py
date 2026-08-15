"""Parameter-alias discovery and validation.

Aliases are based on tensor storage identity, not equal values.  This matters for
tied input/output embeddings: equal copies are not a tied parameter.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class AliasGroup:
    """A canonical parameter name and every name sharing its storage."""

    canonical: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("an alias group must contain at least two members")
        if tuple(sorted(set(self.members))) != self.members:
            raise ValueError("alias members must be unique and sorted")
        if self.canonical != self.members[0]:
            raise ValueError("the lexicographically first member is canonical")

    def to_dict(self) -> dict[str, object]:
        return {"canonical": self.canonical, "members": list(self.members)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AliasGroup:
        canonical = value.get("canonical")
        members = value.get("members")
        if not isinstance(canonical, str) or not isinstance(members, list):
            raise ValueError("malformed alias group")
        if not all(isinstance(item, str) for item in members):
            raise ValueError("alias members must be strings")
        return cls(canonical=canonical, members=tuple(cast(list[str], members)))


def _storage_key(tensor: Tensor) -> tuple[str, int, int, int]:
    # data_ptr is stable for the lifetime of the loaded model.  Include storage
    # offset and byte length so disjoint views of one allocation are not aliases.
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        storage.data_ptr(),
        int(tensor.storage_offset()) * tensor.element_size(),
        int(tensor.numel()) * tensor.element_size(),
    )


def discover_parameter_aliases(model: nn.Module) -> tuple[AliasGroup, ...]:
    """Return deterministic groups for physically shared named parameters."""

    grouped: dict[tuple[str, int, int, int], list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        grouped[_storage_key(parameter)].append(name)
    result = []
    for names in grouped.values():
        members = tuple(sorted(set(names)))
        if len(members) > 1:
            result.append(AliasGroup(canonical=members[0], members=members))
    return tuple(sorted(result, key=lambda group: group.canonical))


def alias_map(groups: Iterable[AliasGroup]) -> dict[str, str]:
    """Map every noncanonical alias name to its canonical parameter name."""

    result: dict[str, str] = {}
    for group in groups:
        for member in group.members:
            previous = result.setdefault(member, group.canonical)
            if previous != group.canonical:
                raise ValueError(f"parameter belongs to multiple alias groups: {member}")
    return result

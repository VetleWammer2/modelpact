"""Finite, deterministic prompt grammar expansion."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemplateGrammar:
    templates: tuple[str, ...]
    variables: Mapping[str, tuple[str, ...]]
    maximum_outputs: int = 100_000

    def expand(self) -> tuple[str, ...]:
        names = tuple(sorted(self.variables))
        domains: Sequence[tuple[str, ...]] = tuple(self.variables[name] for name in names)
        outputs: list[str] = []
        for template in self.templates:
            for values in itertools.product(*domains):
                if len(outputs) >= self.maximum_outputs:
                    raise ValueError(f"grammar exceeds maximum output count {self.maximum_outputs}")
                substitutions = dict(zip(names, values, strict=True))
                try:
                    outputs.append(template.format_map(substitutions))
                except (KeyError, ValueError) as error:
                    raise ValueError(f"invalid template: {template!r}") from error
        # Sorting and deduplication make source mapping order irrelevant.
        return tuple(sorted(set(outputs)))


def finite_cartesian(fields: Mapping[str, Sequence[str]], *, limit: int = 100_000) -> tuple[dict[str, str], ...]:
    names = tuple(sorted(fields))
    outputs: list[dict[str, str]] = []
    for values in itertools.product(*(tuple(fields[name]) for name in names)):
        if len(outputs) >= limit:
            raise ValueError(f"Cartesian generator exceeds limit {limit}")
        outputs.append(dict(zip(names, values, strict=True)))
    return tuple(outputs)


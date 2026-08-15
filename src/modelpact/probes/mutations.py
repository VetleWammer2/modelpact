"""Deterministic local behavioral fuzzing operators."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import StrEnum


class MutationOperator(StrEnum):
    ENTITY_SUBSTITUTION = "entity_substitution"
    NUMBER_SUBSTITUTION = "number_substitution"
    CASING = "casing"
    PUNCTUATION = "punctuation"
    INSTRUCTION_ORDER = "instruction_ordering"
    ROLE_WRAPPER = "role_wrapper"
    DISTRACTOR_CONTEXT = "distractor_context_insertion"
    IRRELEVANT_SENTENCE = "irrelevant_sentence_insertion"
    SYNONYMOUS_TEMPLATE = "synonymous_template_substitution"
    WHITESPACE = "whitespace_formatting"
    COMPLETION_PREFIX = "completion_prefix_perturbation"


@dataclass(frozen=True, slots=True)
class Mutation:
    operator: MutationOperator
    source: str
    mutated: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MutationConfig:
    entities: tuple[str, ...] = ("Aster", "Beryl", "Cyra")
    numbers: tuple[int, ...] = (0, 1, 2, 7, 13, 42)
    role_wrappers: tuple[str, ...] = ("User: {prompt}\nAssistant:", "[INST] {prompt} [/INST]")
    distractors: tuple[str, ...] = ("Context: the sky is blue.", "Note: this sentence is unrelated.")
    synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    completion_prefixes: tuple[str, ...] = ("", "Answer: ", "Response: ")
    maximum_per_prompt: int = 64


_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)")
_ENTITY = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _append(results: list[Mutation], mutation: Mutation, *, limit: int) -> None:
    if mutation.mutated != mutation.source and all(item.mutated != mutation.mutated for item in results):
        if len(results) < limit:
            results.append(mutation)


def mutate_prompt(prompt: str, *, config: MutationConfig = MutationConfig(), seed: int = 0) -> tuple[Mutation, ...]:
    """Apply every configured operator in a stable seeded order."""

    rng = random.Random(seed)
    results: list[Mutation] = []
    entities = list(dict.fromkeys(config.entities))
    rng.shuffle(entities)
    match = _ENTITY.search(prompt)
    if match:
        for entity in entities:
            _append(
                results,
                Mutation(MutationOperator.ENTITY_SUBSTITUTION, prompt, prompt[: match.start()] + entity + prompt[match.end() :]),
                limit=config.maximum_per_prompt,
            )
    number_match = _NUMBER.search(prompt)
    if number_match:
        for number in config.numbers:
            _append(
                results,
                Mutation(MutationOperator.NUMBER_SUBSTITUTION, prompt, prompt[: number_match.start()] + str(number) + prompt[number_match.end() :]),
                limit=config.maximum_per_prompt,
            )
    for casing in (prompt.lower(), prompt.upper(), prompt.swapcase()):
        _append(results, Mutation(MutationOperator.CASING, prompt, casing), limit=config.maximum_per_prompt)
    for punctuation in (prompt.rstrip(" .!?"), prompt.rstrip(" .!?") + "?", prompt.rstrip(" .!?") + "."):
        _append(results, Mutation(MutationOperator.PUNCTUATION, prompt, punctuation), limit=config.maximum_per_prompt)
    instructions = [part.strip() for part in re.split(r"(?:\n+|;\s+)", prompt) if part.strip()]
    if len(instructions) > 1:
        _append(results, Mutation(MutationOperator.INSTRUCTION_ORDER, prompt, "\n".join(reversed(instructions))), limit=config.maximum_per_prompt)
    for wrapper in config.role_wrappers:
        _append(results, Mutation(MutationOperator.ROLE_WRAPPER, prompt, wrapper.format(prompt=prompt)), limit=config.maximum_per_prompt)
    for distractor in config.distractors:
        _append(results, Mutation(MutationOperator.DISTRACTOR_CONTEXT, prompt, f"{distractor}\n{prompt}"), limit=config.maximum_per_prompt)
        _append(results, Mutation(MutationOperator.IRRELEVANT_SENTENCE, prompt, f"{prompt} {distractor}"), limit=config.maximum_per_prompt)
    for source in sorted(config.synonyms):
        if source in prompt:
            for replacement in sorted(config.synonyms[source]):
                _append(results, Mutation(MutationOperator.SYNONYMOUS_TEMPLATE, prompt, prompt.replace(source, replacement)), limit=config.maximum_per_prompt)
    for whitespace in (re.sub(r" +", "  ", prompt), prompt.replace(" ", "\n"), f"  {prompt}  "):
        _append(results, Mutation(MutationOperator.WHITESPACE, prompt, whitespace), limit=config.maximum_per_prompt)
    for prefix in config.completion_prefixes:
        _append(results, Mutation(MutationOperator.COMPLETION_PREFIX, prompt, prompt + "\n" + prefix), limit=config.maximum_per_prompt)
    return tuple(results)


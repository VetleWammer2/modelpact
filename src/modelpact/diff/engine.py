"""Executed differential testing between two trusted local model adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from modelpact.adapters.base import GenerationPolicy, ModelAdapter
from modelpact.diff.activations import concatenate_fingerprints, projected_difference
from modelpact.diff.gradients import gradient_fingerprint
from modelpact.diff.metrics import jensen_shannon, symmetric_kl, top_token_flip_rate
from modelpact.diff.witnesses import DifferenceWitness
from modelpact.probes.minimize import minimize_prompt
from modelpact.probes.mutations import MutationConfig
from modelpact.probes.search import SearchConfig, search_prompts


@dataclass(frozen=True, slots=True)
class DiffConfig:
    divergence_threshold: float = 0.01
    search_budget: int = 256
    generation_max_new_tokens: int = 32
    activation_dimensions: int = 8
    gradient_dimensions: int = 8
    maximum_activation_points: int = 8
    maximum_gradient_modules: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.divergence_threshold < 0 or not torch.isfinite(torch.tensor(self.divergence_threshold)):
            raise ValueError("divergence threshold must be finite and nonnegative")
        if min(self.search_budget, self.generation_max_new_tokens, self.activation_dimensions, self.gradient_dimensions) <= 0:
            raise ValueError("diff budgets and dimensions must be positive")


@dataclass(frozen=True, slots=True)
class DiffExecution:
    witnesses: tuple[DifferenceWitness, ...]
    prompts_evaluated: int
    tokens_processed: int
    wall_seconds: float
    search_budget: int
    threshold: float


def _capture_activations(
    adapter: ModelAdapter,
    model: nn.Module,
    prompt: str,
    *,
    maximum_points: int,
) -> tuple[Tensor, ...]:
    captured: list[Tensor] = []
    handles: list[Any] = []

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, Tensor):
            captured.append(tensor.detach().cpu())

    for point in tuple(adapter.activation_points(model))[:maximum_points]:
        handles.append(point.module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            adapter.forward_logits(model, adapter.tokenizer().batch([prompt]))
    finally:
        for handle in handles:
            handle.remove()
    return tuple(captured)


def _teacher_gradient_fingerprint(
    adapter: ModelAdapter,
    student: nn.Module,
    teacher: nn.Module,
    prompt: str,
    *,
    maximum_modules: int,
    dimensions: int,
    seed: int,
) -> tuple[float, ...]:
    batch = adapter.tokenizer().batch([prompt])
    with torch.no_grad():
        teacher_logits = adapter.forward_logits(teacher, batch).detach()
    student_logits = adapter.forward_logits(student, batch)
    teacher_probabilities = torch.softmax(teacher_logits.to(torch.float64), dim=-1)
    loss = -(teacher_probabilities * torch.log_softmax(student_logits.to(torch.float64), dim=-1)).sum(dim=-1).mean()
    parameters: list[nn.Parameter] = []
    for module in tuple(adapter.patchable_modules(student))[:maximum_modules]:
        for parameter_name in module.parameter_names:
            parameter = getattr(module.module, parameter_name, None)
            if isinstance(parameter, nn.Parameter) and parameter.requires_grad:
                parameters.append(parameter)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True) if parameters else ()
    return gradient_fingerprint(gradients, dimensions=dimensions, seed=seed)


def _evaluate_prompt(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_model: nn.Module,
    prompt: str,
    config: DiffConfig,
    *,
    detailed: bool,
) -> dict[str, object]:
    batch = adapter.tokenizer().batch([prompt])
    with torch.no_grad():
        base_logits = adapter.forward_logits(base_model, batch).detach().cpu()
        target_logits = adapter.forward_logits(target_model, batch).detach().cpu()
    mask = batch.attention_mask.bool().cpu()
    selected_base = base_logits[mask]
    selected_target = target_logits[mask]
    sequence_lengths = mask.sum(dim=1) - 1
    rows = torch.arange(base_logits.shape[0])
    final_base = base_logits[rows, sequence_lengths]
    final_target = target_logits[rows, sequence_lengths]
    skl = float(symmetric_kl(final_base, final_target).mean().item())
    mean_skl = float(symmetric_kl(selected_base, selected_target).mean().item())
    jsd = float(jensen_shannon(selected_base, selected_target).mean().item())
    flip = top_token_flip_rate(selected_base, selected_target)
    result: dict[str, object] = {
        "symmetric_kl": skl,
        "mean_symmetric_kl": mean_skl,
        "jensen_shannon": jsd,
        "top_token_flip_rate": flip,
        "tokens": int(mask.sum().item()),
    }
    if not detailed:
        return result
    policy = GenerationPolicy(mode="greedy", max_new_tokens=config.generation_max_new_tokens, seed=config.seed)
    base_generated = adapter.generate(base_model, batch, policy)[0]
    target_generated = adapter.generate(target_model, batch, policy)[0]
    base_activations = _capture_activations(adapter, base_model, prompt, maximum_points=config.maximum_activation_points)
    target_activations = _capture_activations(adapter, target_model, prompt, maximum_points=config.maximum_activation_points)
    activation_features = concatenate_fingerprints(
        projected_difference(base, target, dimensions=config.activation_dimensions, seed=config.seed + index)
        for index, (base, target) in enumerate(zip(base_activations, target_activations, strict=False))
        if base.shape == target.shape
    )
    gradient_features = _teacher_gradient_fingerprint(
        adapter,
        base_model,
        target_model,
        prompt,
        maximum_modules=config.maximum_gradient_modules,
        dimensions=config.gradient_dimensions,
        seed=config.seed,
    )
    result.update(
        {
            "base_generation": {"token_ids": base_generated.token_ids, "text": base_generated.text},
            "target_generation": {"token_ids": target_generated.token_ids, "text": target_generated.text},
            "generation_changed": float(base_generated.token_ids != target_generated.token_ids),
            "activation_fingerprint": activation_features,
            "gradient_fingerprint": gradient_features,
        }
    )
    return result


def find_difference_witnesses(
    adapter: ModelAdapter,
    base_model: nn.Module,
    target_model: nn.Module,
    seed_prompts: tuple[str, ...],
    *,
    config: DiffConfig = DiffConfig(),
    mutation_config: MutationConfig = MutationConfig(),
) -> DiffExecution:
    """Find scoped, minimized witnesses under an explicit finite search budget."""

    if not seed_prompts:
        raise ValueError("behavioral diff requires at least one seed prompt")
    adapter.prepare(base_model)
    adapter.prepare(target_model)
    started = time.perf_counter()
    cache: dict[str, dict[str, object]] = {}

    def evaluate(prompt: str) -> tuple[float, float, float]:
        metrics = cache.setdefault(prompt, _evaluate_prompt(adapter, base_model, target_model, prompt, config, detailed=False))
        # Novelty and cluster coverage use prompt/token diversity only as search
        # heuristics; they never decide witness validity.
        novelty = len(set(prompt.encode("utf-8"))) / 256.0
        coverage = float(metrics["top_token_flip_rate"])
        return float(metrics["symmetric_kl"]), novelty, coverage

    candidates = search_prompts(
        seed_prompts,
        evaluate,
        config=SearchConfig(budget=config.search_budget, seed=config.seed),
        mutation_config=mutation_config,
    )
    witnesses: list[DifferenceWitness] = []
    token_count = sum(int(cache[item.prompt]["tokens"]) for item in candidates)
    for candidate in candidates:
        metrics = cache[candidate.prompt]
        if float(metrics["symmetric_kl"]) < config.divergence_threshold:
            continue

        def preserves(prompt: str) -> bool:
            return float(_evaluate_prompt(adapter, base_model, target_model, prompt, config, detailed=False)["symmetric_kl"]) >= config.divergence_threshold

        minimized = minimize_prompt(candidate.prompt, preserves)
        detailed = _evaluate_prompt(adapter, base_model, target_model, minimized.minimized, config, detailed=True)
        metric_values = {
            name: float(detailed[name])
            for name in ("symmetric_kl", "jensen_shannon", "top_token_flip_rate", "generation_changed")
        }
        witnesses.append(
            DifferenceWitness.create(
                original_input=candidate.prompt,
                minimized_input=minimized.minimized,
                divergence_metrics=metric_values,
                base_output=detailed["base_generation"],
                target_output=detailed["target_generation"],
                activation_fingerprint=tuple(detailed["activation_fingerprint"]),  # type: ignore[arg-type]
                gradient_fingerprint=tuple(detailed["gradient_fingerprint"]),  # type: ignore[arg-type]
                provenance={
                    "mutation": None if candidate.mutation is None else candidate.mutation.operator.value,
                    "minimization_evaluations": minimized.evaluations,
                    "threshold": config.divergence_threshold,
                    "search_budget": config.search_budget,
                },
            )
        )
    unique = {witness.witness_id: witness for witness in witnesses}
    return DiffExecution(
        witnesses=tuple(unique[key] for key in sorted(unique)),
        prompts_evaluated=len(candidates),
        tokens_processed=token_count,
        wall_seconds=time.perf_counter() - started,
        search_budget=config.search_budget,
        threshold=config.divergence_threshold,
    )

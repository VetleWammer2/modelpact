"""Differentiable compilation objectives declared by Behavior Contract v1."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional

from modelpact.contracts.ast import CompileObjective, ObjectiveType
from modelpact.status import VerificationOutcome


@dataclass(frozen=True, slots=True)
class ObjectiveInputs:
    """Tensor inputs supplied by the trusted model adapter/compiler.

    Keeping model execution outside the objective evaluator makes the numerical
    functions directly testable and prevents contract data from selecting code.
    """

    logits: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    teacher_logits: torch.Tensor | None = None
    base_logits: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    preferred_log_prob: torch.Tensor | None = None
    dispreferred_log_prob: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    reference_hidden_states: torch.Tensor | None = None
    activations: torch.Tensor | None = None
    direction: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    objective_id: str
    objective_type: ObjectiveType
    outcome: VerificationOutcome
    loss: torch.Tensor | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None

    @property
    def supported(self) -> bool:
        return self.outcome is VerificationOutcome.PASS


def _unsupported(spec: CompileObjective, reason: str) -> ObjectiveEvaluation:
    return ObjectiveEvaluation(
        objective_id=spec.id,
        objective_type=spec.type,
        outcome=VerificationOutcome.UNSUPPORTED,
        loss=None,
        reason=reason,
    )


def _inconclusive(spec: CompileObjective, reason: str) -> ObjectiveEvaluation:
    return ObjectiveEvaluation(
        objective_id=spec.id,
        objective_type=spec.type,
        outcome=VerificationOutcome.INCONCLUSIVE,
        loss=None,
        reason=reason,
    )


def _option_float(spec: CompileObjective, name: str, default: float) -> float:
    value = spec.options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _option_bool(spec: CompileObjective, name: str, default: bool) -> bool:
    value = spec.options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    normalized = mask.to(device=values.device, dtype=values.dtype)
    while normalized.ndim > values.ndim:
        normalized = normalized.squeeze(-1)
    if normalized.shape != values.shape:
        try:
            normalized = torch.broadcast_to(normalized, values.shape)
        except RuntimeError as error:
            raise ValueError("token_mask cannot be broadcast to per-token losses") from error
    denominator = normalized.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("token_mask selects no values")
    return (values * normalized).sum() / denominator


def _causal_tensors(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    shifted: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, time, vocabulary] and labels [batch, time]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit and label batch/time dimensions differ")
    if logits.shape[-1] < 2:
        raise ValueError("vocabulary must contain at least two tokens")
    if shifted:
        if logits.shape[1] < 2:
            raise ValueError("causal shifting requires at least two time steps")
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        if mask is not None:
            if mask.shape != (logits.shape[0], logits.shape[1] + 1):
                raise ValueError("token_mask shape does not match unshifted labels")
            mask = mask[:, 1:]
    elif mask is not None and mask.shape != labels.shape:
        raise ValueError("token_mask shape does not match labels")
    return logits, labels, mask


def _cross_entropy(spec: CompileObjective, inputs: ObjectiveInputs) -> ObjectiveEvaluation:
    if inputs.logits is None or inputs.labels is None:
        return _unsupported(spec, "teacher_cross_entropy requires logits and labels")
    try:
        shifted = _option_bool(spec, "causal_shift", True)
        logits, labels, mask = _causal_tensors(
            inputs.logits, inputs.labels, inputs.token_mask, shifted=shifted
        )
        temperature = _option_float(spec, "temperature", 1.0)
        ignore_index_raw = spec.options.get("ignore_index", -100)
        if isinstance(ignore_index_raw, bool) or not isinstance(ignore_index_raw, int):
            raise ValueError("ignore_index must be an integer")
        losses = functional.cross_entropy(
            (logits / temperature).reshape(-1, logits.shape[-1]),
            labels.reshape(-1).to(dtype=torch.long),
            reduction="none",
            ignore_index=ignore_index_raw,
        ).reshape(labels.shape)
        valid = labels.ne(ignore_index_raw)
        effective_mask = valid if mask is None else valid & mask.to(dtype=torch.bool)
        loss = _masked_mean(losses, effective_mask) * spec.weight
    except (RuntimeError, ValueError) as error:
        return _inconclusive(spec, str(error))
    return _finished(spec, loss)


def _kl_objective(
    spec: CompileObjective,
    inputs: ObjectiveInputs,
    *,
    reference: torch.Tensor | None,
    reference_name: str,
) -> ObjectiveEvaluation:
    if inputs.logits is None or reference is None:
        return _unsupported(spec, f"{spec.type.value} requires logits and {reference_name}")
    if inputs.logits.shape != reference.shape or inputs.logits.ndim != 3:
        return _inconclusive(spec, "student and reference logits must have identical [B,T,V] shape")
    try:
        temperature = _option_float(spec, "temperature", 1.0)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        student = inputs.logits
        teacher = reference.to(device=student.device, dtype=student.dtype)
        shifted = _option_bool(spec, "causal_shift", False)
        mask = inputs.token_mask
        if shifted:
            if student.shape[1] < 2:
                raise ValueError("causal shifting requires at least two time steps")
            student = student[:, :-1, :]
            teacher = teacher[:, :-1, :]
            if mask is not None:
                if mask.shape != inputs.logits.shape[:2]:
                    raise ValueError("token_mask shape differs from logit batch/time dimensions")
                mask = mask[:, 1:]
        elif mask is not None and mask.shape != student.shape[:2]:
            raise ValueError("token_mask shape differs from logit batch/time dimensions")
        student_log = functional.log_softmax(student / temperature, dim=-1)
        teacher_log = functional.log_softmax(teacher / temperature, dim=-1)
        teacher_prob = teacher_log.exp()
        per_token = (teacher_prob * (teacher_log - student_log)).sum(dim=-1)
        loss = _masked_mean(per_token, mask) * (temperature**2) * spec.weight
    except (RuntimeError, ValueError) as error:
        return _inconclusive(spec, str(error))
    return _finished(spec, loss)


def _preferred_margin(spec: CompileObjective, inputs: ObjectiveInputs) -> ObjectiveEvaluation:
    preferred = inputs.preferred_log_prob
    dispreferred = inputs.dispreferred_log_prob
    if preferred is None or dispreferred is None:
        return _unsupported(
            spec, "preferred_sequence_margin requires preferred and dispreferred log probabilities"
        )
    if preferred.shape != dispreferred.shape or preferred.numel() == 0:
        reason = "preferred and dispreferred scores must have equal nonempty shape"
        return _inconclusive(spec, reason)
    try:
        margin = _option_float(spec, "margin", 0.0)
        per_item = functional.softplus(margin - (preferred - dispreferred))
        reduction = spec.options.get("reduction", "mean")
        if reduction == "mean":
            loss = per_item.mean()
        elif reduction == "sum":
            loss = per_item.sum()
        else:
            raise ValueError("reduction must be mean or sum")
        loss = loss * spec.weight
    except (RuntimeError, ValueError) as error:
        return _inconclusive(spec, str(error))
    return _finished(spec, loss)


def _hidden_matching(spec: CompileObjective, inputs: ObjectiveInputs) -> ObjectiveEvaluation:
    hidden = inputs.hidden_states
    reference = inputs.reference_hidden_states
    if hidden is None or reference is None:
        reason = "hidden_state_matching requires hidden and reference hidden states"
        return _unsupported(spec, reason)
    if hidden.shape != reference.shape or hidden.numel() == 0:
        return _inconclusive(spec, "hidden-state tensors must have identical nonempty shapes")
    reference = reference.to(device=hidden.device, dtype=hidden.dtype)
    metric = spec.options.get("metric", "mse")
    try:
        if metric == "mse":
            loss = functional.mse_loss(hidden, reference)
        elif metric == "cosine":
            if hidden.shape[-1] < 1:
                raise ValueError("hidden-state feature dimension is empty")
            loss = (1.0 - functional.cosine_similarity(hidden, reference, dim=-1)).mean()
        else:
            raise ValueError("metric must be mse or cosine")
        loss = loss * spec.weight
    except (RuntimeError, ValueError) as error:
        return _inconclusive(spec, str(error))
    return _finished(spec, loss)


def _activation_direction(spec: CompileObjective, inputs: ObjectiveInputs) -> ObjectiveEvaluation:
    activations = inputs.activations
    direction = inputs.direction
    if activations is None or direction is None:
        return _unsupported(spec, "activation_direction requires activations and a direction")
    if activations.ndim < 1 or direction.ndim != 1 or activations.shape[-1] != direction.shape[0]:
        return _inconclusive(spec, "direction must match the activation feature dimension")
    try:
        normalized = functional.normalize(
            direction.to(device=activations.device, dtype=activations.dtype), dim=0
        )
        projection = torch.matmul(activations, normalized)
        if _option_bool(spec, "absolute", False):
            projection = projection.abs()
        threshold = _option_float(spec, "minimum_projection", 0.0)
        loss = functional.relu(threshold - projection).mean() * spec.weight
    except (RuntimeError, ValueError) as error:
        return _inconclusive(spec, str(error))
    return _finished(spec, loss, extra={"mean_projection": float(projection.detach().mean().cpu())})


def _finished(
    spec: CompileObjective,
    loss: torch.Tensor,
    *,
    extra: Mapping[str, float] | None = None,
) -> ObjectiveEvaluation:
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
        return _inconclusive(spec, "objective produced a non-scalar or non-finite loss")
    metrics = {"loss": float(loss.detach().cpu())}
    metrics.update(extra or {})
    return ObjectiveEvaluation(
        objective_id=spec.id,
        objective_type=spec.type,
        outcome=VerificationOutcome.PASS,
        loss=loss,
        metrics=metrics,
    )


def evaluate_objective(spec: CompileObjective, inputs: ObjectiveInputs) -> ObjectiveEvaluation:
    """Evaluate one declared objective without dispatching arbitrary code."""

    if spec.type is ObjectiveType.TEACHER_CROSS_ENTROPY:
        return _cross_entropy(spec, inputs)
    if spec.type is ObjectiveType.TEACHER_KL:
        return _kl_objective(
            spec,
            inputs,
            reference=inputs.teacher_logits,
            reference_name="teacher_logits",
        )
    if spec.type is ObjectiveType.BASE_KL:
        return _kl_objective(
            spec,
            inputs,
            reference=inputs.base_logits,
            reference_name="base_logits",
        )
    if spec.type is ObjectiveType.PREFERRED_SEQUENCE_MARGIN:
        return _preferred_margin(spec, inputs)
    if spec.type is ObjectiveType.HIDDEN_STATE_MATCHING:
        return _hidden_matching(spec, inputs)
    if spec.type is ObjectiveType.ACTIVATION_DIRECTION:
        return _activation_direction(spec, inputs)
    return _unsupported(spec, f"objective type {spec.type!s} is not implemented")


__all__ = ["ObjectiveEvaluation", "ObjectiveInputs", "evaluate_objective"]

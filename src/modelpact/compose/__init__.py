"""Behavioral composition and semantic-merge primitives."""

from modelpact.compose.closure import (
    CompositionResult,
    ContractMargin,
    MarginKind,
    PatchOperand,
    VerificationReport,
    additive_compose,
    verify_contract_closure,
)
from modelpact.compose.interactions import contract_margin_interaction

__all__ = [
    "CompositionResult",
    "ContractMargin",
    "MarginKind",
    "PatchOperand",
    "VerificationReport",
    "additive_compose",
    "contract_margin_interaction",
    "verify_contract_closure",
]

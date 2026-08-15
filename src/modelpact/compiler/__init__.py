"""Contract-guided low-rank-plus-sparse patch compilation."""

from modelpact.compiler.optimize import compile_low_rank_patch
from modelpact.compiler.result import CompilationResult

__all__ = ["CompilationResult", "compile_low_rank_patch"]

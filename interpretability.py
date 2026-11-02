"""Component Interpretability & Loadings Analysis for SpectraCompress.

Provides rigorous, governance-compliant interpretation of decomposition components:
  - Top-m features by absolute loading with preserved signs.
  - Deterministic svd_flip-style sign normalization on a strictly isolated copy (transformer is never mutated).
  - Cosine stability attachment via component_cosine_stability across CV folds or perturbation references.
  - Heuristic spectral ratio flag detecting clustered eigenvalues / potential rotation ambiguity.
  - Tracking of whether features were standardized prior to decomposition.
  - Explicit embedded governance warning:
    "Loadings describe variance structure, not causation. Sign and rotation are not unique;
     components with close eigenvalues are unstable. A low cosine stability can also result
     from component order swaps between folds with close eigenvalues."
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

from error import InputValidation, NotFittedError
from stability_diagnostics import component_cosine_stability

if TYPE_CHECKING:
    from decomposition import DecompositionTransformer

logger = logging.getLogger(__name__)

DEFAULT_CAVEAT_TEXT = (
    "Loadings describe variance structure, not causation. "
    "Sign and rotation are not unique; components with close eigenvalues are unstable. "
    "A low cosine stability can also result from component order swaps between folds with close eigenvalues."
)


@dataclass(frozen=True)
class InterpretabilityConfig:
    """Configuration for component loadings reporting and diagnostics.

    Fields:
        top_m: Number of highest-magnitude features to extract per component.
        stability_threshold: Minimum cosine similarity threshold to declare a component stable (default: 0.90).
        rotation_ambiguity_ratio: Eigenvalue ratio threshold (lambda_i / lambda_{i+1}) below which
                                  components are flagged by the heuristic as rotation-ambiguous (default: 1.05).
        causation_caveat: Standard governance caveat text.
    """
    top_m: int = 5
    stability_threshold: float = 0.90
    rotation_ambiguity_ratio: float = 1.05
    causation_caveat: str = DEFAULT_CAVEAT_TEXT

    def __post_init__(self):
        if self.top_m < 1:
            raise InputValidation(message=f"top_m must be >= 1, got {self.top_m}")
        if not (0.0 < self.stability_threshold <= 1.0):
            raise InputValidation(
                message=f"stability_threshold must be in (0, 1], got {self.stability_threshold}"
            )
        if self.rotation_ambiguity_ratio <= 1.0:
            raise InputValidation(
                message=f"rotation_ambiguity_ratio must be > 1.0, got {self.rotation_ambiguity_ratio}"
            )


@dataclass
class FeatureLoading:
    """Individual feature loading on a principal component."""
    feature_index: int
    feature_name: str
    loading: float
    abs_loading: float


@dataclass
class ComponentSummary:
    """Summary of a single decomposition component."""
    component_index: int
    explained_variance_ratio: float
    singular_value: float
    top_features: List[FeatureLoading]
    cosine_stability: Optional[float]
    is_stable: Optional[bool]
    adjacent_eigenvalue_ratio: Optional[float]
    is_rotation_ambiguous: bool


@dataclass
class ComponentLoadingsReport:
    """Complete interpretability and loadings report for a fitted transformer."""
    n_components: int
    n_features: int
    top_m: int
    features_standardized: Union[bool, str]
    stability_status: str  # "COMPUTED" or "NOT COMPUTED"
    caveat: str
    components: List[ComponentSummary]

    def to_dict(self) -> Dict[str, Any]:
        """Converts report into a clean, JSON-serializable dictionary."""
        return asdict(self)


def component_loadings_report(
    transformer: "DecompositionTransformer",
    feature_names: Optional[Sequence[str]] = None,
    top_m: int = 5,
    *,
    stability_reference: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    config: Optional[InterpretabilityConfig] = None,
    features_standardized: Optional[Union[bool, str]] = None,
) -> ComponentLoadingsReport:
    """Generates a loadings and stability report for a fitted DecompositionTransformer.

    Enforces deterministic sign convention by making the largest absolute loading in each
    component positive. Does NOT mutate the transformer components; operates strictly on a copy.

    Spectral Ambiguity Note (G1):
        The condition lambda_i / lambda_{i+1} <= tol is a HEURISTIC ratio flag indicating clustered
        variance, not the formal Davis-Kahan theorem (which depends on the absolute spectral gap
        and perturbation operator norm). For the last retained component (i = k - 1), if an un-retained
        trailing singular value exists in transformer.singular_values_, the forward ratio against it
        is evaluated; otherwise, only the backward ratio (lambda_{k-2} / lambda_{k-1}) is evaluated.
        If k == 1, rotation ambiguity is False. Division-by-zero is strictly guarded.

    Args:
        transformer: Fitted DecompositionTransformer instance (or loaded artifact).
        feature_names: Optional sequence of strings naming each feature.
        top_m: Number of highest-magnitude features to extract per component.
        stability_reference: Optional reference component matrix or sequence of fold matrices
                             to compute cosine stability against. If absent, stability is marked "NOT COMPUTED".
        config: Optional InterpretabilityConfig governing thresholds and caveats.
        features_standardized: Optional override or check whether features were scaled to unit variance.

    Returns:
        ComponentLoadingsReport containing component summaries and diagnostics.

    Raises:
        NotFittedError: If transformer has not been fitted.
        InputValidation: If feature_names length does not match n_features_in_.
    """
    if getattr(transformer, "components_", None) is None:
        raise NotFittedError(message="Transformer is not fitted. Cannot generate loadings report.")

    if config is None:
        config = InterpretabilityConfig(top_m=top_m)
    elif top_m != 5 and config.top_m == 5:
        config = InterpretabilityConfig(
            top_m=top_m,
            stability_threshold=config.stability_threshold,
            rotation_ambiguity_ratio=config.rotation_ambiguity_ratio,
            causation_caveat=config.causation_caveat,
        )

    effective_k, d = transformer.components_.shape
    if transformer.n_features_in_ is not None and transformer.n_features_in_ != d:
        d = transformer.n_features_in_

    # Determine whether features were standardized (G3)
    if features_standardized is not None:
        standardized_flag = features_standardized
    elif transformer.centerer is not None and hasattr(transformer.centerer, "with_std"):
        standardized_flag = bool(transformer.centerer.with_std)
    else:
        standardized_flag = "unknown"

    # Validate feature names
    if feature_names is not None:
        if len(feature_names) != d:
            raise InputValidation(
                message=f"Length of feature_names ({len(feature_names)}) does not match n_features ({d})"
            )
        resolved_names = list(feature_names)
    else:
        resolved_names = [f"feature_{j}" for j in range(d)]

    effective_m = min(config.top_m, d)

    if effective_k == 0:
        return ComponentLoadingsReport(
            n_components=0,
            n_features=d,
            top_m=effective_m,
            features_standardized=standardized_flag,
            stability_status="NOT COMPUTED",
            caveat=config.causation_caveat,
            components=[],
        )

    # 1. Deterministic svd_flip sign normalization strictly on an isolated COPY
    W_norm = transformer.components_.copy()
    max_abs_cols = np.argmax(np.abs(W_norm), axis=1)
    signs = np.sign(W_norm[range(effective_k), max_abs_cols])
    signs[signs == 0] = 1.0
    W_norm *= signs[:, np.newaxis]

    # 2. Stability attachment via component_cosine_stability
    if stability_reference is None:
        stability_status = "NOT COMPUTED"
        stabilities: List[Optional[float]] = [None] * effective_k
        stable_flags: List[Optional[bool]] = [None] * effective_k
    else:
        stability_status = "COMPUTED"
        if isinstance(stability_reference, np.ndarray):
            ref_list = [stability_reference]
        else:
            ref_list = list(stability_reference)

        fold_sims = []
        for ref in ref_list:
            if not isinstance(ref, np.ndarray) or ref.ndim != 2:
                continue
            ref_k = ref.shape[0]
            eval_k = min(effective_k, ref_k)
            sim = component_cosine_stability(W_norm[:eval_k], ref[:eval_k])
            if eval_k < effective_k:
                full_sim = np.full(effective_k, np.nan, dtype=np.float64)
                full_sim[:eval_k] = sim
                fold_sims.append(full_sim)
            else:
                fold_sims.append(sim[:effective_k])

        if len(fold_sims) > 0:
            mean_sim = np.nanmean(fold_sims, axis=0)
            stabilities = [float(s) if np.isfinite(s) else None for s in mean_sim]
            stable_flags = [
                bool(s >= config.stability_threshold) if s is not None else None
                for s in stabilities
            ]
        else:
            stability_status = "NOT COMPUTED"
            stabilities = [None] * effective_k
            stable_flags = [None] * effective_k

    # 3. Heuristic adjacent eigenvalue ratio evaluation (G1)
    lambdas = getattr(transformer, "explained_variance_", None)
    if lambdas is None or len(lambdas) < effective_k:
        if transformer.singular_values_ is not None and len(transformer.singular_values_) >= effective_k:
            lambdas = transformer.singular_values_ ** 2
        else:
            lambdas = np.ones(effective_k, dtype=np.float64)

    # Check for additional trailing singular values to evaluate the last retained component
    trailing_lambda = None
    if (
        transformer.singular_values_ is not None
        and len(transformer.singular_values_) > effective_k
    ):
        trailing_lambda = float(transformer.singular_values_[effective_k] ** 2)

    component_reports: List[ComponentSummary] = []
    for i in range(effective_k):
        ratios: List[float] = []

        # Backward ratio: lambda_{i-1} / lambda_i (with zero-division guard)
        if i > 0:
            denom = float(lambdas[i])
            if denom > 1e-15:
                ratios.append(float(lambdas[i - 1]) / denom)

        # Forward ratio: lambda_i / lambda_{i+1} (with zero-division guard)
        if i < effective_k - 1:
            denom = float(lambdas[i + 1])
            if denom > 1e-15:
                ratios.append(float(lambdas[i]) / denom)
        elif trailing_lambda is not None and trailing_lambda > 1e-15:
            # Last component evaluated against the next unretained singular value
            ratios.append(float(lambdas[i]) / trailing_lambda)

        if len(ratios) > 0:
            min_ratio = float(min(ratios))
            is_rotation_ambiguous = bool(min_ratio <= config.rotation_ambiguity_ratio)
        else:
            min_ratio = None
            is_rotation_ambiguous = False

        # Extract top-m features by absolute loading
        weights = W_norm[i, :]
        abs_weights = np.abs(weights)
        top_indices = np.argsort(abs_weights)[::-1][:effective_m]

        top_feature_entries = [
            FeatureLoading(
                feature_index=int(idx),
                feature_name=resolved_names[idx],
                loading=float(weights[idx]),
                abs_loading=float(abs_weights[idx]),
            )
            for idx in top_indices
        ]

        exp_var_ratio = (
            float(transformer.explained_variance_ratio_[i])
            if getattr(transformer, "explained_variance_ratio_", None) is not None
            and len(transformer.explained_variance_ratio_) > i
            else 0.0
        )

        sing_val = (
            float(transformer.singular_values_[i])
            if getattr(transformer, "singular_values_", None) is not None
            and len(transformer.singular_values_) > i
            else 0.0
        )

        component_reports.append(
            ComponentSummary(
                component_index=i,
                explained_variance_ratio=exp_var_ratio,
                singular_value=sing_val,
                top_features=top_feature_entries,
                cosine_stability=stabilities[i],
                is_stable=stable_flags[i],
                adjacent_eigenvalue_ratio=min_ratio,
                is_rotation_ambiguous=is_rotation_ambiguous,
            )
        )

    return ComponentLoadingsReport(
        n_components=effective_k,
        n_features=d,
        top_m=effective_m,
        features_standardized=standardized_flag,
        stability_status=stability_status,
        caveat=config.causation_caveat,
        components=component_reports,
    )

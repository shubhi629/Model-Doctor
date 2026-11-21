"""Mitigation layer and decision engine for ModelDoctor.

Translates Model 2 trustworthiness scores into actionable decisions:
    - TRUST  : prediction meets verified reliability criteria
    - REVIEW : elevated risk or domain anomalies; human examination recommended
    - REJECT : prediction does not meet reliability standard; model abstains

Threshold rules:
- Chosen ONLY from Out-Of-Fold (OOF) calibration predictions.
- Never chosen from Model 2's predictions on its own training rows.
- Never chosen from the test split.
- reject_threshold < trust_threshold always.
- If no threshold below trust_threshold exhibits >= reject_min_error (default 30%)
  across at least reject_min_rows (default 30), reject_threshold is set to None.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Risk level communicated with each decision (a fixed label of the decision, not a new score).
RISK_LEVELS: Dict[str, str] = {"TRUST": "LOW", "REVIEW": "MEDIUM", "REJECT": "HIGH"}

# Which explanation group each warning code belongs to (used by the API and dashboard).
WARNING_CATEGORIES: Dict[str, str] = {
    "OUT_OF_RANGE": "out_of_distribution",
    "UNSEEN_CATEGORY": "out_of_distribution",
    "OOD_DISTANCE": "out_of_distribution",
    "UNSUPPORTED_ESTIMATE": "out_of_distribution",
    "ANOMALOUS_INPUT": "anomaly",
    "MISSING_FIELD": "missing_information",
    "MISSING_FIELDS": "missing_information",
    "MODEL_DISAGREEMENT": "model_disagreement",
    "LOW_CONFIDENCE": "uncertainty",
    "HIGH_ENTROPY": "uncertainty",
    "LOW_MARGIN": "uncertainty",
    "LOW_TRUSTWORTHINESS": "low_trustworthiness",
}
DOMAIN_ANOMALY_FLAGS = ("out_of_range", "unseen_category")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def compute_decision_band_metadata(
    oof_trustworthiness: np.ndarray,
    y_correct: np.ndarray,
    trust_threshold: float,
    reject_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Compute rich JSON-serializable metadata for each decision band.
    """
    oof_trust = np.asarray(oof_trustworthiness, dtype=float)
    y_arr = np.asarray(y_correct, dtype=int)
    n_total = len(oof_trust)
    note_text = (
        "These rates are estimated on the calibration split, where the thresholds "
        "were tuned, so they are optimistic. Confirm them on held-out data."
    )

    bands: List[Dict[str, Any]] = []

    # 1. TRUST band
    trust_mask = oof_trust >= trust_threshold
    n_trust = int(np.sum(trust_mask))
    err_trust = int(np.sum((1 - y_arr)[trust_mask]))
    bands.append({
        "name": "TRUST",
        "threshold_range": f">= {trust_threshold:.3f}",
        "count": n_trust,
        "share": round(float(n_trust / n_total), 4) if n_total > 0 else 0.0,
        "observed_error_rate": round(float(err_trust / n_trust), 4) if n_trust > 0 else 0.0,
        "estimate_source": "calibration_oof",
        "note": note_text,
    })

    # 2. REVIEW band
    if reject_threshold is not None:
        review_mask = (oof_trust >= reject_threshold) & (oof_trust < trust_threshold)
        range_str = f"[{reject_threshold:.3f}, {trust_threshold:.3f})"
    else:
        review_mask = oof_trust < trust_threshold
        range_str = f"< {trust_threshold:.3f}"
    n_review = int(np.sum(review_mask))
    err_review = int(np.sum((1 - y_arr)[review_mask]))
    bands.append({
        "name": "REVIEW",
        "threshold_range": range_str,
        "count": n_review,
        "share": round(float(n_review / n_total), 4) if n_total > 0 else 0.0,
        "observed_error_rate": round(float(err_review / n_review), 4) if n_review > 0 else 0.0,
        "estimate_source": "calibration_oof",
        "note": note_text,
    })

    # 3. REJECT band
    if reject_threshold is not None:
        reject_mask = oof_trust < reject_threshold
        n_reject = int(np.sum(reject_mask))
        err_reject = int(np.sum((1 - y_arr)[reject_mask]))
        range_str = f"< {reject_threshold:.3f}"
        rate_val = round(float(err_reject / n_reject), 4) if n_reject > 0 else 0.0
    else:
        n_reject = 0
        range_str = "disabled"
        rate_val = 0.0
    bands.append({
        "name": "REJECT",
        "threshold_range": range_str,
        "count": n_reject,
        "share": round(float(n_reject / n_total), 4) if n_total > 0 else 0.0,
        "observed_error_rate": rate_val,
        "estimate_source": "calibration_oof",
        "note": note_text,
    })

    return bands


def tune_mitigation_thresholds(
    oof_trustworthiness: np.ndarray,
    y_correct: np.ndarray,
    target_risk: float = 0.05,
    min_coverage: float = 0.30,
    reject_threshold: Optional[float] = None,
    reject_min_error: float = 0.30,
    reject_min_rows: int = 30,
    reject_relative_factor: float = 2.0,
    reject_error_cap: float = 0.50,
) -> Dict[str, Any]:
    """
    Select optimal trust and reject cutoffs from Out-Of-Fold (OOF) calibration probabilities.

    Parameters
    ----------
    oof_trustworthiness : np.ndarray
        Out-of-fold correctness probabilities P(correct) on the calibration split.
    y_correct : np.ndarray
        Binary true correctness: 1 = correct, 0 = incorrect.
    target_risk : float, default=0.05
        Maximum acceptable error rate among TRUST predictions (5%).
    min_coverage : float, default=0.30
        Minimum fraction of predictions that must be accepted (30%).
    reject_threshold : float, optional
        Manual override for reject threshold. If None, computed data-driven from OOF.
    reject_min_error : float, default=0.30
        Minimum error rate required in the REJECT band.
    reject_min_rows : int, default=30
        Minimum sample size required in the REJECT band.
    reject_relative_factor : float, default=2.0
        The REJECT band must also show at least this multiple of the overall
        calibration error rate (capped at ``reject_error_cap``). Without it, a model
        whose overall error already exceeds ``reject_min_error`` would send nearly
        every non-TRUST prediction to REJECT, even when most of them are correct.
    reject_error_cap : float, default=0.50
        Upper bound on the relative requirement: a band where predictions are more
        likely wrong than right always qualifies.

    Returns
    -------
    dict
        JSON-serializable threshold configuration.
    """
    oof_trust = np.asarray(oof_trustworthiness, dtype=np.float64)
    y_arr = np.asarray(y_correct, dtype=int)
    n_total = len(oof_trust)

    if n_total == 0:
        band_meta = compute_decision_band_metadata(oof_trust, y_arr, 0.85, None)
        return {
            "trust_threshold": 0.85,
            "reject_threshold": None,
            "target_risk": target_risk,
            "min_coverage": min_coverage,
            "target_risk_met": False,
            "message": "No calibration samples available to tune thresholds.",
            "calibration_coverage": 0.0,
            "calibration_selective_risk": 0.0,
            "band_metadata": band_meta,
        }

    # Grid of candidate thresholds between 0.50 and 0.99
    grid = np.unique(np.clip(np.round(oof_trust, 4), 0.50, 0.99))
    if len(grid) == 0:
        grid = np.linspace(0.50, 0.95, 46)
    grid = np.sort(grid)

    target_risk_met = False
    chosen_trust = None
    chosen_risk = 0.0
    chosen_coverage = 0.0

    # Scan ascending: find lowest threshold meeting selective_risk <= target_risk & coverage >= min_coverage
    for tau in grid:
        accepted = oof_trust >= tau
        n_acc = int(np.sum(accepted))
        if n_acc == 0:
            continue

        coverage = float(n_acc / n_total)
        if coverage < min_coverage:
            continue

        risk = float(np.mean(1 - y_arr[accepted]))
        if risk <= target_risk:
            chosen_trust = float(tau)
            chosen_risk = risk
            chosen_coverage = coverage
            target_risk_met = True
            break

    # If target risk cannot be met with min_coverage, pick threshold meeting min_coverage with lowest risk
    if not target_risk_met:
        candidates_min_cov = []
        for tau in grid:
            accepted = oof_trust >= tau
            n_acc = int(np.sum(accepted))
            if n_acc == 0:
                continue
            coverage = float(n_acc / n_total)
            if coverage >= min_coverage:
                risk = float(np.mean(1 - y_arr[accepted]))
                candidates_min_cov.append((tau, risk, coverage))

        if candidates_min_cov:
            candidates_min_cov.sort(key=lambda x: (x[1], -x[2]))
            chosen_trust, chosen_risk, chosen_coverage = candidates_min_cov[0]
            msg_trust = (
                f"Target selective risk of {target_risk*100:.1f}% could not be achieved with >= {min_coverage*100:.1f}% coverage. "
                f"Best achieved risk is {chosen_risk*100:.1f}% at {chosen_coverage*100:.1f}% coverage (threshold: {chosen_trust:.3f})."
            )
        else:
            chosen_trust = 0.85
            accepted = oof_trust >= chosen_trust
            chosen_coverage = float(np.mean(accepted))
            chosen_risk = float(np.mean(1 - y_arr[accepted])) if np.sum(accepted) > 0 else 0.0
            msg_trust = f"Could not meet minimum coverage of {min_coverage*100:.1f}%. Defaulting to threshold 0.85."
    else:
        msg_trust = (
            f"Target selective risk of {target_risk*100:.1f}% successfully met at threshold {chosen_trust:.3f} "
            f"with {chosen_coverage*100:.1f}% coverage."
        )

    # -------------------------------------------------------------------------
    # Data-Driven Reject Threshold Selection
    # -------------------------------------------------------------------------
    chosen_reject: Optional[float] = None
    msg_reject = ""

    if reject_threshold is not None:
        # Manual override
        if reject_threshold < chosen_trust:
            chosen_reject = float(reject_threshold)
            msg_reject = f"Manual reject threshold set to {chosen_reject:.3f}."
        else:
            chosen_reject = max(0.10, round(chosen_trust - 0.05, 3))
            msg_reject = f"Manual reject threshold adjusted below trust threshold to {chosen_reject:.3f}."
    else:
        # Data-driven: find highest value tau < chosen_trust such that
        # observed error rate for trustworthiness < tau is >= the required error
        # (max of reject_min_error and a multiple of the base error, capped) and
        # has at least reject_min_rows.
        base_error = float(np.mean(1 - y_arr))
        required_error = max(reject_min_error, min(reject_error_cap, reject_relative_factor * base_error))
        candidate_rejects = np.unique(np.clip(np.round(oof_trust, 4), 0.10, chosen_trust - 0.001))
        # Scan descending (highest first)
        candidate_rejects = np.sort(candidate_rejects)[::-1]

        for tau in candidate_rejects:
            rejected = oof_trust < tau
            n_rej = int(np.sum(rejected))
            if n_rej >= reject_min_rows:
                err_rate = float(np.mean(1 - y_arr[rejected]))
                if err_rate >= required_error:
                    chosen_reject = float(tau)
                    msg_reject = (
                        f"Data-driven reject threshold set at {chosen_reject:.3f} "
                        f"({n_rej} rows, observed error rate {err_rate*100:.1f}% >= {required_error*100:.1f}%)."
                    )
                    break

        if chosen_reject is None:
            msg_reject = (
                f"No threshold below trust threshold ({chosen_trust:.3f}) has an observed error rate "
                f">= {required_error*100:.1f}% across at least {reject_min_rows} rows; REJECT band disabled."
            )

    full_message = f"{msg_trust} {msg_reject}".strip()
    band_meta = compute_decision_band_metadata(oof_trust, y_arr, chosen_trust, chosen_reject)

    return {
        "trust_threshold": round(chosen_trust, 4),
        "reject_threshold": round(chosen_reject, 4) if chosen_reject is not None else None,
        "target_risk": float(target_risk),
        "min_coverage": float(min_coverage),
        "target_risk_met": bool(target_risk_met),
        "message": full_message,
        "calibration_coverage": round(chosen_coverage, 4),
        "calibration_selective_risk": round(chosen_risk, 4),
        "band_metadata": band_meta,
    }


def generate_warnings(
    record_flags: List[Dict[str, Any]],
    features: Optional[Dict[str, Any]] = None,
    decision: Optional[str] = None,
    num_classes: int = 2,
) -> List[Dict[str, str]]:
    """
    Convert raw diagnostic flags and feature signals into concise human-readable warnings
    accompanied by machine-readable reason codes.

    If decision is REVIEW or REJECT and no feature flags fired, returns fallback warning
    LOW_TRUSTWORTHINESS.
    For TRUST with no flags, returns empty list.
    For binary targets (num_classes=2), merges redundant HIGH_ENTROPY, LOW_MARGIN,
    and LOW_CONFIDENCE into a single LOW_CONFIDENCE warning.
    """
    warnings: List[Dict[str, str]] = []

    missing_fields = []
    out_of_range_fields = []
    unseen_cats = []

    for flag in (record_flags or []):
        f_type = flag.get("type")
        if f_type == "missing_feature":
            missing_fields.append(flag.get("feature"))
        elif f_type == "out_of_range":
            out_of_range_fields.append(flag.get("feature"))
        elif f_type == "unseen_category":
            unseen_cats.append(f"{flag.get('feature')} ('{flag.get('value')}')")
        elif f_type == "high_knn_distance":
            pct = flag.get("percentile", 95.0)
            warnings.append({
                "code": "OOD_DISTANCE",
                "message": f"the input is unusually far from the training data ({pct:.0f}th percentile distance)",
            })
        elif f_type == "high_isolation_score":
            warnings.append({
                "code": "ANOMALOUS_INPUT",
                "message": "input flagged as anomalous by isolation forest",
            })
        elif f_type == "high_bootstrap_disagreement":
            disagree = flag.get("disagreement", 0.3)
            warnings.append({
                "code": "MODEL_DISAGREEMENT",
                "message": f"model copies disagree strongly ({int(disagree*100)}% disagreement)",
            })
        elif f_type == "high_entropy":
            warnings.append({
                "code": "HIGH_ENTROPY",
                "message": "prediction distribution is highly uncertain",
            })
        elif f_type == "low_margin":
            warnings.append({
                "code": "LOW_MARGIN",
                "message": "small margin between top predicted classes",
            })
        elif f_type == "outside_reliability_support":
            feats = ", ".join(flag.get("features", []))
            warnings.append({
                "code": "UNSUPPORTED_ESTIMATE",
                "message": (
                    "reliability signals are outside the range the reliability model was trained on "
                    f"({feats}), so the trustworthiness estimate is an extrapolation"
                ),
            })

    # Grouped missing fields
    if len(missing_fields) == 1:
        warnings.append({
            "code": "MISSING_FIELD",
            "message": f"field '{missing_fields[0]}' is missing",
        })
    elif len(missing_fields) > 1:
        warnings.append({
            "code": "MISSING_FIELDS",
            "message": f"{len(missing_fields)} fields are missing ({', '.join(missing_fields)})",
        })

    # Out of range fields
    for col in out_of_range_fields:
        warnings.append({
            "code": "OUT_OF_RANGE",
            "message": f"'{col}' is outside the training range",
        })

    # Unseen categories
    for cat_desc in unseen_cats:
        warnings.append({
            "code": "UNSEEN_CATEGORY",
            "message": f"unseen categorical value for {cat_desc}",
        })

    # Low claimed confidence check
    if features is not None:
        conf = features.get("max_prob")
        if conf is not None and conf < 0.60:
            warnings.append({
                "code": "LOW_CONFIDENCE",
                "message": f"primary model confidence is low ({conf*100:.1f}%)",
            })

    # For binary targets, HIGH_ENTROPY, LOW_MARGIN and LOW_CONFIDENCE are deterministic
    # functions of the same number (max_prob). Merge them into ONE LOW_CONFIDENCE warning.
    if num_classes == 2:
        conf_codes = {"HIGH_ENTROPY", "LOW_MARGIN", "LOW_CONFIDENCE"}
        matching = [w for w in warnings if w["code"] in conf_codes]
        if len(matching) >= 1:
            conf_val = None
            if features is not None and "max_prob" in features:
                conf_val = features["max_prob"]
            # Filter out redundant items
            warnings = [w for w in warnings if w["code"] not in conf_codes]
            # Replace with a single LOW_CONFIDENCE warning
            msg = f"primary model confidence is low ({conf_val*100:.1f}%)" if conf_val is not None else "primary model confidence is low"
            warnings.append({
                "code": "LOW_CONFIDENCE",
                "message": msg,
            })

    # Fallback for non-TRUST decisions when no flags fired
    if decision in ["REVIEW", "REJECT"] and len(warnings) == 0:
        warnings.append({
            "code": "LOW_TRUSTWORTHINESS",
            "message": "Estimated trustworthiness is below the trust level for this prediction.",
        })

    return warnings


class MitigationEngine:
    """
    Mitigation decision engine holding thresholds and generating final actions.
    Can be serialized with joblib.
    """

    def __init__(
        self,
        trust_threshold: float = 0.85,
        reject_threshold: Optional[float] = 0.50,
        target_risk: float = 0.05,
        min_coverage: float = 0.30,
        target_risk_met: bool = True,
        message: Optional[str] = None,
        force_review_on_anomalies: bool = False,
        calibration_coverage: float = 0.0,
        calibration_selective_risk: float = 0.0,
        num_classes: int = 2,
        band_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        if reject_threshold is not None and reject_threshold >= trust_threshold:
            raise ValueError(
                f"reject_threshold ({reject_threshold}) must be strictly lower than trust_threshold ({trust_threshold})."
            )

        self.trust_threshold = float(trust_threshold)
        self.reject_threshold = float(reject_threshold) if reject_threshold is not None else None
        self.target_risk = float(target_risk)
        self.min_coverage = float(min_coverage)
        self.target_risk_met = bool(target_risk_met)
        self.message = message or ""
        self.force_review_on_anomalies = bool(force_review_on_anomalies)
        self.calibration_coverage = float(calibration_coverage)
        self.calibration_selective_risk = float(calibration_selective_risk)
        self.num_classes = int(num_classes)
        self.band_metadata = band_metadata or []

    def decide(
        self,
        trustworthiness: float,
        record_flags: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Compute mitigation decision: TRUST, REVIEW, or REJECT.
        Handles reject_threshold=None (returning only TRUST or REVIEW).
        """
        score = float(trustworthiness)

        if score >= self.trust_threshold:
            decision = "TRUST"
        elif self.reject_threshold is not None and score < self.reject_threshold:
            decision = "REJECT"
        else:
            decision = "REVIEW"

        # Mandatory: never auto-TRUST when Model 2 had to extrapolate beyond its training data
        if decision == "TRUST" and any(
            f.get("type") == "outside_reliability_support" for f in (record_flags or [])
        ):
            decision = "REVIEW"

        # Optional override: force at least REVIEW when domain anomalies occur
        if self.force_review_on_anomalies and decision == "TRUST" and record_flags:
            has_domain_anomaly = any(
                f.get("type") in DOMAIN_ANOMALY_FLAGS
                for f in record_flags
            )
            if has_domain_anomaly:
                decision = "REVIEW"

        return decision

    def evaluate_record(
        self,
        trustworthiness: float,
        record_flags: Optional[List[Dict[str, Any]]] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single record and return decision + warnings.
        """
        flags = record_flags or []
        decision = self.decide(trustworthiness, flags)
        warnings = generate_warnings(
            flags,
            features,
            decision=decision,
            num_classes=self.num_classes,
        )

        return {
            "trustworthiness": round(float(trustworthiness), 4),
            "decision": decision,
            "warnings": warnings,
        }

    def explain(
        self,
        trustworthiness: float,
        decision: str,
        warnings: List[Dict[str, str]],
        reliability_mode: str = "model2",
    ) -> Dict[str, Any]:
        """
        Plain-English account of a decision, built only from the thresholds and the
        warnings that produced it.

        Returns {"risk_level", "explanation" (one-sentence summary), "reasons" (list)}.
        """
        t, trust_t = _pct(trustworthiness), _pct(self.trust_threshold)
        codes = {w["code"] for w in warnings}
        reasons: List[str] = []

        if decision == "TRUST":
            reasons.append(f"Trustworthiness {t} is at or above the trust threshold {trust_t}.")
            summary = "Accepted: this prediction meets the reliability criteria for automatic use."
        elif decision == "REVIEW":
            if trustworthiness >= self.trust_threshold:
                if "UNSUPPORTED_ESTIMATE" in codes:
                    reasons.append(
                        f"Trustworthiness {t} meets the trust threshold {trust_t}, but the reliability model had to "
                        "extrapolate beyond the data it was trained on, so the estimate cannot be relied on."
                    )
                else:
                    reasons.append(
                        f"Trustworthiness {t} meets the trust threshold {trust_t}, but the input has out-of-range or "
                        "unseen values and this model is configured to send such inputs to review."
                    )
            elif self.reject_threshold is None:
                reasons.append(f"Trustworthiness {t} is below the trust threshold {trust_t}.")
            else:
                reasons.append(
                    f"Trustworthiness {t} is below the trust threshold {trust_t} but not below the reject "
                    f"threshold {_pct(self.reject_threshold)}."
                )
            summary = "Needs human review: a person should verify this prediction before it is used."
        else:
            reject_t = _pct(self.reject_threshold) if self.reject_threshold is not None else trust_t
            reasons.append(f"Trustworthiness {t} is below the reject threshold {reject_t}.")
            summary = (
                "Rejected: ModelDoctor abstains because this prediction is too likely to be wrong to act on."
            )

        for w in warnings:
            if w["code"] == "LOW_TRUSTWORTHINESS":
                continue
            msg = w["message"].rstrip(".")
            reasons.append(msg[:1].upper() + msg[1:] + ".")

        if reliability_mode != "model2":
            reasons.append(
                "Trustworthiness equals Model 1's calibrated confidence here: the calibration split had too few "
                "errors to train the separate reliability model."
            )
        return {"risk_level": RISK_LEVELS[decision], "explanation": summary, "reasons": reasons}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return {
            "trust_threshold": self.trust_threshold,
            "reject_threshold": self.reject_threshold,
            "target_risk": self.target_risk,
            "min_coverage": self.min_coverage,
            "target_risk_met": self.target_risk_met,
            "message": self.message,
            "force_review_on_anomalies": self.force_review_on_anomalies,
            "calibration_coverage": self.calibration_coverage,
            "calibration_selective_risk": self.calibration_selective_risk,
            "num_classes": self.num_classes,
            "band_metadata": self.band_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MitigationEngine":
        """Deserialize from configuration dictionary."""
        return cls(**data)

"""Test-split evaluation of ModelDoctor against the confidence-threshold baseline.

CRITICAL RULES:
- Everything here is computed on the TEST split only, AFTER all models and
  thresholds were fixed. Nothing in this module is used to tune anything.
- Every comparison includes the "No filtering" reference and the
  confidence-threshold baseline evaluated at the SAME coverage as ModelDoctor
  (fair comparison, spec section 8).
- All outputs are JSON-serializable (lists / dicts of Python floats).
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.reliability.model2 import compute_brier_score, compute_ece


def _safe_auc(is_incorrect: np.ndarray, risk_score: np.ndarray) -> Optional[float]:
    """AUROC for ranking incorrect predictions above correct ones (None if undefined)."""
    if len(np.unique(is_incorrect)) < 2:
        return None
    return round(float(roc_auc_score(is_incorrect, risk_score)), 4)


def _safe_auprc(is_incorrect: np.ndarray, risk_score: np.ndarray) -> Optional[float]:
    """Average precision for detecting incorrect predictions (None if undefined)."""
    if len(np.unique(is_incorrect)) < 2:
        return None
    return round(float(average_precision_score(is_incorrect, risk_score)), 4)


def selective_metrics(correct: np.ndarray, accepted: np.ndarray) -> Dict[str, Any]:
    """Coverage and selective risk (error among accepted) for a boolean acceptance mask."""
    correct = np.asarray(correct, dtype=int)
    accepted = np.asarray(accepted, dtype=bool)
    n = len(correct)
    n_acc = int(accepted.sum())
    risk = float(np.mean(1 - correct[accepted])) if n_acc > 0 else 0.0
    return {
        "coverage": round(n_acc / n, 4) if n else 0.0,
        "selective_risk": round(risk, 4),
        "accepted": n_acc,
        "total": int(n),
    }


def accept_top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Accept exactly the k rows with the highest score (stable tie-breaking)."""
    score = np.asarray(score, dtype=float)
    mask = np.zeros(len(score), dtype=bool)
    if k <= 0:
        return mask
    order = np.argsort(-score, kind="mergesort")
    mask[order[:k]] = True
    return mask


def error_detection_metrics(correct: np.ndarray, flagged: np.ndarray) -> Dict[str, Any]:
    """Precision / recall of flagging (not accepting) a prediction as a detector of errors."""
    correct = np.asarray(correct, dtype=int)
    flagged = np.asarray(flagged, dtype=bool)
    incorrect = correct == 0
    tp = int(np.sum(flagged & incorrect))
    fp = int(np.sum(flagged & ~incorrect))
    fn = int(np.sum(~flagged & incorrect))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "flagged": int(flagged.sum()),
        "errors_caught": tp,
        "errors_total": int(incorrect.sum()),
    }


def risk_coverage_curve(
    correct: np.ndarray,
    score: np.ndarray,
    n_points: int = 50,
) -> List[Dict[str, float]]:
    """
    Selective risk at evenly spaced coverage levels, accepting the highest-scored rows first.
    """
    correct = np.asarray(correct, dtype=int)
    n = len(correct)
    if n == 0:
        return []
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    cum_errors = np.cumsum(1 - correct[order])
    ks = np.unique(np.clip(np.round(np.linspace(1, n, n_points)).astype(int), 1, n))
    return [
        {
            "coverage": round(float(k / n), 4),
            "selective_risk": round(float(cum_errors[k - 1] / k), 4),
        }
        for k in ks
    ]


def area_under_risk_coverage(correct: np.ndarray, score: np.ndarray) -> float:
    """AURC: mean selective risk over all coverage levels (lower is better)."""
    correct = np.asarray(correct, dtype=int)
    n = len(correct)
    if n == 0:
        return 0.0
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    cum_errors = np.cumsum(1 - correct[order])
    risks = cum_errors / np.arange(1, n + 1)
    return round(float(np.mean(risks)), 4)


def calibration_curve(
    correct: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Equal-width reliability diagram bins: mean predicted P(correct) vs observed accuracy."""
    correct = np.asarray(correct, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob >= lo) & ((prob <= hi) if i == n_bins - 1 else (prob < hi))
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        out.append({
            "bin_low": round(float(lo), 2),
            "bin_high": round(float(hi), 2),
            "count": cnt,
            "mean_predicted": round(float(prob[mask].mean()), 4),
            "observed_accuracy": round(float(correct[mask].mean()), 4),
        })
    return out


def evaluate_system(
    correct: np.ndarray,
    confidence: np.ndarray,
    trustworthiness: np.ndarray,
    decisions: List[str],
    reliability_mode: str = "model2",
) -> Dict[str, Any]:
    """
    Full test-split evaluation: ModelDoctor (TRUST decisions from thresholds tuned on the
    calibration split) vs the confidence baseline at the same coverage, plus curves.

    Parameters
    ----------
    correct : array of int
        1 if Model 1's test prediction was correct, else 0.
    confidence : array of float
        Model 1 max predicted probability (baseline score).
    trustworthiness : array of float
        ModelDoctor P(correct) estimates.
    decisions : list[str]
        TRUST / REVIEW / REJECT for each test row.
    reliability_mode : str
        "model2" or "confidence_fallback" (Model 2 had insufficient signal).
    """
    correct = np.asarray(correct, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    trust = np.asarray(trustworthiness, dtype=float)
    decisions_arr = np.asarray(decisions)
    n = len(correct)
    is_incorrect = 1 - correct

    md_accepted = decisions_arr == "TRUST"
    n_acc = int(md_accepted.sum())
    baseline_accepted = accept_top_k(confidence, n_acc)

    no_filter = selective_metrics(correct, np.ones(n, dtype=bool))
    md = selective_metrics(correct, md_accepted)
    base = selective_metrics(correct, baseline_accepted)

    comparison = [
        {"method": "No filtering", **no_filter},
        {"method": "Confidence baseline", **base},
        {"method": "ModelDoctor", **md},
    ]

    auroc = {
        "confidence_baseline": _safe_auc(is_incorrect, 1.0 - confidence),
        "modeldoctor": _safe_auc(is_incorrect, 1.0 - trust),
    }
    auprc = {
        "confidence_baseline": _safe_auprc(is_incorrect, 1.0 - confidence),
        "modeldoctor": _safe_auprc(is_incorrect, 1.0 - trust),
        "random": round(float(is_incorrect.mean()), 4) if n else 0.0,
    }
    aurc = {
        "confidence_baseline": area_under_risk_coverage(correct, confidence),
        "modeldoctor": area_under_risk_coverage(correct, trust),
    }

    decision_distribution = []
    for name in ["TRUST", "REVIEW", "REJECT"]:
        mask = decisions_arr == name
        cnt = int(mask.sum())
        decision_distribution.append({
            "decision": name,
            "count": cnt,
            "share": round(cnt / n, 4) if n else 0.0,
            "observed_error_rate": round(float(is_incorrect[mask].mean()), 4) if cnt else None,
            "mean_trustworthiness": round(float(trust[mask].mean()), 4) if cnt else None,
        })

    risk_delta = round(base["selective_risk"] - md["selective_risk"], 4)
    if n_acc == 0:
        verdict = "ModelDoctor accepted no test predictions; no comparison possible at zero coverage."
    elif reliability_mode != "model2":
        verdict = (
            "Model 2 had insufficient error signal, so ModelDoctor fell back to Model 1 confidence; "
            "it is identical to the baseline by construction."
        )
    elif risk_delta > 0.005:
        verdict = (
            f"ModelDoctor lowers error among accepted predictions by {risk_delta * 100:.1f} points "
            f"vs the confidence baseline at the same {md['coverage'] * 100:.1f}% coverage."
        )
    elif risk_delta < -0.005:
        verdict = (
            f"ModelDoctor is worse than the confidence baseline by {-risk_delta * 100:.1f} points "
            f"at the same {md['coverage'] * 100:.1f}% coverage on this dataset."
        )
    else:
        verdict = "ModelDoctor matches the confidence baseline at the same coverage on this dataset."

    return {
        "n_test": int(n),
        "overall_error_rate": no_filter["selective_risk"],
        "comparison": comparison,
        "selective_risk_improvement": risk_delta,
        "verdict": verdict,
        "auroc": auroc,
        "auprc": auprc,
        "aurc": aurc,
        "error_detection": {
            "modeldoctor": error_detection_metrics(correct, ~md_accepted),
            "confidence_baseline": error_detection_metrics(correct, ~baseline_accepted),
        },
        "calibration": {
            "modeldoctor": {
                "ece": compute_ece(correct, trust),
                "brier": compute_brier_score(correct, trust),
            },
            "confidence_baseline": {
                "ece": compute_ece(correct, confidence),
                "brier": compute_brier_score(correct, confidence),
            },
        },
        "decision_distribution": decision_distribution,
        "risk_coverage_curve": {
            "modeldoctor": risk_coverage_curve(correct, trust),
            "baseline": risk_coverage_curve(correct, confidence),
        },
        "calibration_curve": {
            "modeldoctor": calibration_curve(correct, trust),
            "baseline": calibration_curve(correct, confidence),
        },
    }

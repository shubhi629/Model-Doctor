"""Update PROJECT_SPEC.md with Task 5c design choices and findings."""
from pathlib import Path

spec_path = Path(__file__).resolve().parents[2] / "project_spec.md"
with open(spec_path, "r", encoding="utf-8") as f:
    content = f.read()

# Normalize newlines
content = content.replace("\r\n", "\n")

old_72 = """### 7.2 Reliability features (model-agnostic)

| Feature | Meaning | How |
|---|---|---|
| Max probability | Claimed confidence | `predict_proba` max |
| Entropy | Spread of probabilities | `-sum(p * log(p))` |
| Margin | Top prob minus second prob | sorted probabilities |
| Bootstrap disagreement | Do resampled copies of the model disagree? | Train several copies on bootstrap samples of the train split, measure prediction variance / vote disagreement |
| kNN distance | How far the input is from training data | Mean distance to k nearest train samples (preprocessed space) |
| Isolation Forest score | Out-of-distribution indicator | Fit on train features only |
| Missing-value count | Input quality | Count of missing fields in the raw input (computed from raw record BEFORE preprocessing/imputation) |
| Out-of-range count | Features outside training min/max | Compare raw numeric values to train ranges (computed from raw record BEFORE preprocessing/imputation) |
| Ensemble variance (optional) | Tree disagreement | Only for Random Forest / Gradient Boosting |

The distance, Isolation Forest, missing-count, and out-of-range signals depend only on the **input data**, so they are fully model-independent. Missing-value count and out-of-range count must be calculated strictly from the raw input before any imputation or scaling."""

new_72 = """### 7.2 Reliability features (model-agnostic)

| Feature | Meaning | How |
|---|---|---|
| Max probability | Claimed confidence | `predict_proba` max |
| Entropy | Spread of probabilities | `-sum(p * log(p))` (normalized by log(num_classes)) |
| Margin | Top prob minus second prob | sorted probabilities |
| Bootstrap disagreement | Internal ensemble uncertainty | $1 - \\text{fraction agreeing with bootstrap ensemble majority}$ |
| Bootstrap std | Standard deviation across copies | Std dev of predicted positive class probabilities |
| kNN distance | How far the input is from training data | Mean distance to k nearest train samples (preprocessed space) |
| Isolation Forest score | Out-of-distribution indicator | Fit on train features only |
| Missing-value count | Input quality | Count of missing fields in the raw input (computed from raw record BEFORE preprocessing/imputation) |
| Missing-value fraction | Normalized input quality | Fraction of missing fields from raw input |
| Out-of-range count | Features outside training min/max | Compare raw numeric values to train ranges (computed from raw record BEFORE preprocessing/imputation) |
| Unseen category count | Categorical novelty | Count of categorical levels not present in train split |

The distance, Isolation Forest, missing-count, and out-of-range signals depend only on the **input data**, so they are fully model-independent. Missing-value count and out-of-range count must be calculated strictly from the raw input before any imputation or scaling.

**Collinearity Guard for Binary Classification:**
For binary classification targets ($K=2$), `entropy` and `margin` are exact monotonic transformations of `max_prob` (confidence). Including them introduces exact collinearity in linear models and redundant splits in tree models. They are excluded by default for binary targets (`exclude_redundant_binary_features=True`) and retained for multiclass targets ($K \\ge 3$). In addition, `predicted_class` is excluded by default (`exclude_predicted_class=True`) to prevent Model 2 from memorizing base class priors."""

assert old_72 in content, "Could not find old_72"
content = content.replace(old_72, new_72)

old_73 = """### 7.3 Model 2 (reliability model)
- **Target:** `correct = (prediction == true_label)` on the calibration split.
- **Input:** the reliability features above (never the true label).
- **Algorithm:** start with Logistic Regression or Gradient Boosting; keep it simple and calibrated.
- **Output:** probability that Model 1's prediction is correct (trustworthiness).
- Use cross-validation *within the calibration split* when training and evaluating Model 2 internally, so its own score is not optimistic.
- **Threshold Tuning:** Thresholds for TRUST/REVIEW/REJECT must be tuned on out-of-fold (cross-validated) Model 2 predictions on the calibration split, NOT on Model 2's predictions for its own training rows.
- Handle the case where Model 1 is almost always correct (very few "incorrect" examples): warn the user that the reliability model has too little signal."""

new_73 = """### 7.3 Model 2 (reliability model)
- **Target:** `correct = (prediction == true_label)` on the calibration split.
- **Input:** the reliability features above (never the true label).
- **Architecture and Probability Calibration:**
  - Unweighted training (`class_weight=None`) is strictly enforced so that Model 2 outputs honest, well-calibrated probabilities rather than reweighted/balanced probabilities.
  - Candidates evaluated via cross-validation: Logistic Regression and Gradient Boosting (wrapped in `CalibratedClassifierCV(method="sigmoid")` to prevent overfitting on calibration data).
  - Out-of-Fold (OOF) cross-validation (5-fold Stratified K-Fold) is computed strictly within the calibration split.
  - Final Model 2 is fit on the full calibration split after candidate evaluation.
- **Sample Size Guard:** Requires at least 30 incorrect predictions (`n_incorrect >= 30`) to train Model 2; if fewer errors are observed, the system gracefully enters `insufficient_signal` status and falls back to Model 1 confidence.
- **Output:** probability that Model 1's prediction is correct (trustworthiness).
- **Threshold Tuning:** Thresholds for TRUST/REVIEW/REJECT must be tuned on out-of-fold (cross-validated) Model 2 predictions on the calibration split, NOT on Model 2's predictions for its own training rows, and NEVER on the test split."""

assert old_73 in content, "Could not find old_73"
content = content.replace(old_73, new_73)

old_75 = """### 7.5 Mitigation layer

| Decision | Meaning | Action |
|---|---|---|
| **TRUST** | Meets required reliability | Accept and display the prediction |
| **REVIEW** | Warning signs or elevated error risk | Flag for human examination |
| **REJECT** | Does not meet required reliability | Abstain from a definitive prediction |

Default thresholds (tuned on out-of-fold cross-validated Model 2 predictions on calibration data, user-configurable):
- Trustworthiness ≥ 0.85 → TRUST
- 0.60 to 0.85 → REVIEW
- Below 0.60 → REJECT

Better approach: choose thresholds to hit a target (e.g. "error among accepted predictions ≤ 5%") on out-of-fold calibration predictions, instead of fixing numbers by hand.

**Warnings:** generate human-readable reasons from the feature values (e.g. "income outside training range", "2 missing fields", "model disagreement high", "input far from training data")."""

new_75 = """### 7.5 Mitigation layer

| Decision | Meaning | Action |
|---|---|---|
| **TRUST** | Meets required reliability | Accept and display the prediction |
| **REVIEW** | Warning signs or elevated error risk | Flag for human examination |
| **REJECT** | Does not meet required reliability | Abstain from a definitive prediction |

**Data-Driven Thresholds:**
- **Tuned on Calibration OOF Only:** Thresholds are determined exclusively from out-of-fold cross-validated Model 2 probabilities on the calibration split.
- **Trust Threshold ($\\tau_{\\text{trust}}$):** Chosen to meet the user's target selective risk (e.g. error among accepted predictions $\\le 5\%$) while satisfying minimum coverage ($\\ge 30\%$). If target risk cannot be met, the threshold achieving lowest risk at $\\ge 30\%$ coverage is selected.
- **Reject Threshold ($\\tau_{\\text{reject}}$):** Chosen via a data-driven scan descending below $\\tau_{\\text{trust}}$ to find the highest threshold where observed error rate is $\\ge 30\%$ across at least 30 samples. If no threshold meets these criteria, `reject_threshold` is set to `None` and the REJECT band is disabled (predictions below $\\tau_{\\text{trust}}$ are routed to REVIEW).
- **Decision Band Metadata:** Returned with each configuration, reporting band name, threshold range, sample count, coverage share, observed error rate, source (`"calibration_oof"`), and an explicit disclaimer note.
- **Warning Merging:** For binary targets ($K=2$), redundant confidence-related flags (`HIGH_ENTROPY`, `LOW_MARGIN`, `LOW_CONFIDENCE`) are merged into a single `LOW_CONFIDENCE` warning. For multiclass targets, individual warnings are preserved."""

assert old_75 in content, "Could not find old_75"
content = content.replace(old_75, new_75)

findings_section = """

---

## 18. Findings So Far (Task 5 / Task 5c Diagnostics)

1. **Synthetic Clean Dataset Baseline vs Model 2:**
   - On the synthetic `sample_data/loans.csv` dataset, Model 1 (Random Forest / Logistic Regression) is well calibrated inside the training split.
   - Primary model confidence-only error-detection AUROC on the calibration split is **0.7405**, whereas Model 2 (trained with additional signals) achieves **0.7319**.
   - **Reasoning:** In an i.i.d. synthetic dataset with no real-world domain shifts or covariate noise, geometric features (kNN distance, Isolation Forest) offer minimal incremental lift over calibrated probability confidence. This is reported honestly per the project charter.

2. **Model 2 Probability Distortion and Recalibration:**
   - Initially, Model 2 was trained with balanced class weighting (`class_weight="balanced"`), causing severe probability overconfidence (ECE 0.2890), where rows in the REJECT band (< 0.50) actually had an observed error rate of only 24.8% (75% correct).
   - Switching to unweighted training (`class_weight=None`) and sigmoid calibration for Gradient Boosting (`CalibratedClassifierCV(method="sigmoid")`) restored honest probabilities, dropping ECE to **0.0173** and Brier score from 0.2018 to **0.1039**.

3. **Collinearity in Binary Signals:**
   - For binary targets, entropy and margin are exact functions of `max_prob`. Suppressing them removed redundant noise and improved Model 2 CV error AUROC from 0.7287 to **0.7319**.
   - In the mitigation layer, merging redundant confidence warnings into a single `LOW_CONFIDENCE` warning prevents user confusion while preserving distinct warnings for multiclass classification.

4. **Internal Ensemble Disagreement:**
   - Measuring bootstrap disagreement relative to the bootstrap ensemble's own majority prediction rather than against Model 1's calibrated probability eliminated the 0.36 artificial discrepancy previously observed, lowering mean disagreement to **0.0760**.
"""

if "## 18. Findings So Far" not in content:
    content = content.rstrip() + findings_section

with open(spec_path, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(content)

print("PROJECT_SPEC.md successfully updated!")

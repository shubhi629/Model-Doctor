# ModelDoctor: Intelligent Detection and Mitigation of Unreliable ML Predictions

> **Instruction to the coding agent:** Read this file fully before starting any task. Re-read it whenever a task touches data splits, reliability features, or the API. If anything here is unclear or conflicts with a request, ask before coding.

---

## 1. Project Idea

A normal ML model returns a prediction for every input, even when that prediction is likely wrong. Overall accuracy (e.g. 90%) does not tell you *which* predictions to doubt.

**ModelDoctor** adds a second model on top of the first. Its only job is to estimate, for each individual prediction, **the probability that the prediction is correct**, and then recommend an action.

- **Model 1 (the predictor):** automatically selected and trained on the user's dataset. Produces the predicted output and a confidence score.
- **Model 2 (the reliability model, "the doctor"):** learns from Model 1's past mistakes on unseen data. Estimates how trustworthy each new prediction is.
- **Mitigation layer:** converts the trustworthiness estimate into **TRUST**, **REVIEW**, or **REJECT**.

One-line summary: *ModelDoctor does not make predictions better; it tells you when not to believe them.*

### Core Research Question

Can a secondary reliability model identify incorrect ML predictions more effectively than a conventional confidence threshold, while maintaining useful prediction coverage?

The project's value depends on demonstrating **measurable improvement over the confidence-threshold baseline**. If ModelDoctor only matches the baseline on some datasets, report that honestly.

---

## 2. Scope

**In scope**
- Tabular datasets uploaded as CSV
- Binary and multiclass **classification** only (Regression is strictly out of scope for this version; upload of continuous numeric targets must return a clear error)
- Automatic model selection from a candidate pool of classification algorithms
- Reliability estimation, mitigation decisions, evaluation against baseline, dashboard

**Out of scope**
- Regression (strictly out of scope for this version; continuous numeric targets are rejected)
- Images, text, time series
- Deep learning models
- Multi-user authentication (a simple session system is enough)

**Dataset constraints**
- Minimum rows: 500 (needed for a meaningful calibration split)
- Maximum upload size: configurable, default 50 MB
- Reject datasets with no valid target column, or a target with only one class

---

## 3. End-to-End Workflow

### Phase A: Setup (once per dataset)

1. **Upload and profile.** User uploads a CSV and selects the target column. System reports rows, columns, types, missing values, class balance, and detected task type (classification or regression).
2. **Three-way split.** Split into train, calibration, and test (see section 4).
3. **Layer 1: automatic model selection.** Cross-validate candidate algorithms on the *train split only*, build a leaderboard, pick the best, retrain it. This is **Model 1**.
4. **Build reliability data.** Model 1 predicts on the *calibration split* (data it never trained on). For each row, record reliability features and a label: was the prediction correct?
5. **Train Model 2.** Learn to predict "prediction is correct" from the reliability features.
6. **Set thresholds.** Using calibration data, choose probability cut-offs for TRUST, REVIEW, REJECT.
7. **Evaluate.** Run the full system on the *test split*. Compare ModelDoctor against the confidence-threshold baseline.
8. **Save everything** under the session ID.

### Phase B: Everyday use (per new record)

The true outcome is **unknown** at this point.

1. User enters a new record (form) or uploads a batch CSV.
2. Model 1 predicts and outputs confidence.
3. Reliability features are extracted.
4. Model 2 estimates trustworthiness (probability the prediction is correct).
5. Mitigation layer returns TRUST / REVIEW / REJECT with warnings explaining why.

Example output:

```
Prediction:        No default
Model confidence:  97%      <- what Model 1 claims
Trustworthiness:   68%      <- what ModelDoctor estimates
Decision:          REVIEW
Warnings:          income outside training range, 2 missing fields, models disagree
```

### Phase C: Monitoring (later)

When real outcomes become available, the user submits them through the feedback endpoint. The system compares estimated reliability with what actually happened, tracks coverage and decision distribution, flags degradation or drift, and supports retraining.

---

## 4. Data Splits (CRITICAL)

| Split | Size | Used for | Must NOT be used for |
|---|---|---|---|
| Train | 60% | Training Model 1, cross-validation for candidate model selection, calibrating Model 1 probabilities (e.g. CalibratedClassifierCV with internal CV on train split) | Training Model 2 |
| Calibration | 20% | Generating Model 1 predictions to train Model 2; tuning thresholds on out-of-fold (CV) Model 2 predictions | Training Model 1, calibrating Model 1 |
| Test | 20% | Final evaluation of the whole system only | Anything else, ever |

Rules:
- Use **stratified** splitting for classification.
- Use a fixed random seed for reproducibility.
- Fit the preprocessing pipeline on the **train split only**, then apply it to the others.
- Calibrating Model 1's probabilities must happen inside the train split only (e.g. `CalibratedClassifierCV` with internal CV). The calibration split is strictly reserved for Model 2.
- If Model 2 were trained on Model 1's predictions for its own training data, Model 1 would look near-perfect and Model 2 would learn nothing. Never do this.

---

## 5. Hard Rules (Never Break)

1. **No data leakage** between train, calibration, and test splits.
2. **Model 2 must never receive the true label as an input feature.** True labels are only used to create the "correct / incorrect" training target.
3. **Reliability features must be model-agnostic.** They must work regardless of which algorithm Model 1 turned out to be. Model-specific signals are optional extras only when available.
4. **Missing-value count and out-of-range features must be computed from the RAW input record**, before any preprocessing or imputation.
5. **Always compare against the confidence-threshold baseline.** Every evaluation output includes both.
6. **Thresholds must be tuned on out-of-fold (cross-validated) Model 2 predictions on the calibration split**, never on Model 2's predictions for its own training rows, and never on test data.
7. **Do not report test results from anything tuned on the test set.**
8. **Classification only:** Continuous numeric targets must be rejected with a clear error saying only classification targets are supported in this version.
9. Validate all inputs at prediction time and return clear errors (unknown columns, wrong types, missing required fields).

---

## 6. Layer 1: Automatic Model Selection (Model 1)

### Steps
1. **Profile** the dataset: rows, features, missing values, categorical vs numeric, class imbalance, target type.
2. **Detect task:** categorical or low-cardinality target means classification; continuous numeric target means regression.
3. **Preprocess** with a scikit-learn `Pipeline` / `ColumnTransformer`:
   - Numeric: median imputation, scaling
   - Categorical: most-frequent imputation, one-hot encoding (handle unseen categories safely)
4. **Candidate pool (classification):**

| Model | Class |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=6)` |
| Random Forest | `RandomForestClassifier(n_estimators=200)` |
| XGBoost | `XGBClassifier(eval_metric="logloss")` |
| KNN (optional) | `KNeighborsClassifier` |

   Regression equivalents are added in the regression phase.
5. **Cross-validate** every candidate on the *same folds* of the train split. Scoring: weighted F1 (classification), R² / MAE (regression). For imbalanced data, use class weighting where supported.
6. **Select the best**, retrain on the full train split, and **calibrate** probabilities (e.g. `CalibratedClassifierCV`) so confidence is comparable across algorithms.
7. Return the **leaderboard** (model name, CV score, training time) for the dashboard.

Be honest in documentation: the "intelligence" here is trying several models and choosing by cross-validation.

---

## 7. Layer 2: ModelDoctor

### 7.1 Confidence (classification)
Confidence is the model's claimed certainty: the maximum predicted class probability.

```python
probs = model.predict_proba(X)
confidence = probs.max(axis=1)
```

Confidence is what the model *claims*, not necessarily the truth. It is both the **baseline** to beat and one **input feature** for Model 2.

### 7.2 Reliability features (model-agnostic)

| Feature | Meaning | How |
|---|---|---|
| Max probability | Claimed confidence | `predict_proba` max |
| Entropy | Spread of probabilities | `-sum(p * log(p))` (normalized by log(num_classes)) |
| Margin | Top prob minus second prob | sorted probabilities |
| Bootstrap disagreement | Internal ensemble uncertainty | $1 - \text{fraction agreeing with bootstrap ensemble majority}$ |
| Bootstrap std | Standard deviation across copies | Std dev of predicted positive class probabilities |
| kNN distance | How far the input is from training data | Mean distance to k nearest train samples (preprocessed space) |
| Isolation Forest score | Out-of-distribution indicator | Fit on train features only |
| Missing-value count | Input quality | Count of missing fields in the raw input (computed from raw record BEFORE preprocessing/imputation) |
| Missing-value fraction | Normalized input quality | Fraction of missing fields from raw input |
| Out-of-range count | Features outside training min/max | Compare raw numeric values to train ranges (computed from raw record BEFORE preprocessing/imputation) |
| Unseen category count | Categorical novelty | Count of categorical levels not present in train split |

The distance, Isolation Forest, missing-count, and out-of-range signals depend only on the **input data**, so they are fully model-independent. Missing-value count and out-of-range count must be calculated strictly from the raw input before any imputation or scaling.

**Collinearity Guard for Binary Classification:**
For binary classification targets ($K=2$), `entropy` and `margin` are exact monotonic transformations of `max_prob` (confidence). Including them introduces exact collinearity in linear models and redundant splits in tree models. They are excluded by default for binary targets (`exclude_redundant_binary_features=True`) and retained for multiclass targets ($K \ge 3$). In addition, `predicted_class` is excluded by default (`exclude_predicted_class=True`) to prevent Model 2 from memorizing base class priors.

### 7.3 Model 2 (reliability model)
- **Target:** `correct = (prediction == true_label)` on the calibration split.
- **Input:** the reliability features above (never the true label).
- **Architecture and Probability Calibration:**
  - Unweighted training (`class_weight=None`) is strictly enforced so that Model 2 outputs honest, well-calibrated probabilities rather than reweighted/balanced probabilities.
  - Candidates evaluated via cross-validation: Logistic Regression and Gradient Boosting (wrapped in `CalibratedClassifierCV(method="sigmoid")` to prevent overfitting on calibration data).
  - Out-of-Fold (OOF) cross-validation (5-fold Stratified K-Fold) is computed strictly within the calibration split.
  - Final Model 2 is fit on the full calibration split after candidate evaluation.
- **Sample Size Guard:** Requires at least 30 incorrect predictions (`n_incorrect >= 30`) to train Model 2; if fewer errors are observed, the system gracefully enters `insufficient_signal` status and falls back to Model 1 confidence.
- **Output:** probability that Model 1's prediction is correct (trustworthiness).
- **Threshold Tuning:** Thresholds for TRUST/REVIEW/REJECT must be tuned on out-of-fold (cross-validated) Model 2 predictions on the calibration split, NOT on Model 2's predictions for its own training rows, and NEVER on the test split.

### 7.4 Regression variant (Deprecated / Out of Scope)
*Note: Regression is strictly out of scope for this version. The system only supports classification.*

### 7.5 Mitigation layer

| Decision | Meaning | Action |
|---|---|---|
| **TRUST** | Meets required reliability | Accept and display the prediction |
| **REVIEW** | Warning signs or elevated error risk | Flag for human examination |
| **REJECT** | Does not meet required reliability | Abstain from a definitive prediction |

**Data-Driven Thresholds:**
- **Tuned on Calibration OOF Only:** Thresholds are determined exclusively from out-of-fold cross-validated Model 2 probabilities on the calibration split.
- **Trust Threshold ($\tau_{\text{trust}}$):** Chosen to meet the user's target selective risk (e.g. error among accepted predictions $\le 5\%$) while satisfying minimum coverage ($\ge 30\%$). If target risk cannot be met, the threshold achieving lowest risk at $\ge 30\%$ coverage is selected.
- **Reject Threshold ($\tau_{\text{reject}}$):** Chosen via a data-driven scan descending below $\tau_{\text{trust}}$ to find the highest threshold where observed error rate is $\ge \max(30\%, \min(50\%, 2 \times \text{overall calibration error}))$ across at least 30 samples. The relative term stops a model whose overall error already exceeds 30% from sending nearly every non-TRUST prediction to REJECT. If no threshold meets these criteria, `reject_threshold` is set to `None` and the REJECT band is disabled (predictions below $\tau_{\text{trust}}$ are routed to REVIEW).
- **Decision Band Metadata:** Returned with each configuration, reporting band name, threshold range, sample count, coverage share, observed error rate, source (`"calibration_oof"`), and an explicit disclaimer note.
- **Warning Merging:** For binary targets ($K=2$), redundant confidence-related flags (`HIGH_ENTROPY`, `LOW_MARGIN`, `LOW_CONFIDENCE`) are merged into a single `LOW_CONFIDENCE` warning. For multiclass targets, individual warnings are preserved.

---

## 8. Evaluation

Run on the **test split**. Always compare ModelDoctor against the **confidence-threshold baseline** and a **no-filtering** reference.

| Metric | Question it answers |
|---|---|
| Error-detection precision and recall | How well are incorrect predictions identified? |
| AUROC / AUPRC | How well does the reliability score separate correct from incorrect? |
| Selective risk | Error rate among accepted (TRUST) predictions |
| Coverage | Percentage of predictions still accepted |
| Calibration (ECE, reliability curve) | Do estimated probabilities match observed correctness? |
| Risk-coverage curve | Error vs coverage at all thresholds, ModelDoctor vs baseline |

Expected comparison table (illustrative):

| Method | Coverage | Error among accepted |
|---|---|---|
| No filtering | 100% | 14% |
| Confidence baseline | 70% | 9% |
| ModelDoctor | 70% | 6% |

**Fair comparison:** compare at the *same coverage* (or same error level), not at arbitrary thresholds.

Useful experiments:
- Run ModelDoctor with each algorithm as Model 1; report where the reliability layer helps most (expected: poorly calibrated models like Decision Tree, Naive Bayes).
- Run on 3 to 4 datasets (e.g. loan default, a medical dataset, Titanic, a regression set later).

---

## 9. Tech Stack

**Backend:** Python, FastAPI, Uvicorn, Pydantic, pandas, numpy, scikit-learn, XGBoost, joblib, python-multipart, pytest

**Frontend:** React (Vite), React Router, Axios or TanStack Query (polling), Recharts, Tailwind CSS

**Communication:** REST JSON API. All chart data is returned as **JSON**, not images. React renders charts.

---

## 10. Project Structure

```
modeldoctor/
├── PROJECT_SPEC.md
├── README.md
├── sample_data/                 # real CSVs for testing (loan, titanic, ...)
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, router registration
│   │   ├── api/                 # route modules (datasets, train, report, predict, feedback)
│   │   ├── schemas/             # Pydantic request/response models
│   │   └── core/                # config, session manager, background tasks, errors
│   ├── ml/
│   │   ├── data/                # loading, profiling, task detection, preprocessing, splitting
│   │   ├── models/              # candidate pool, cross-validation, leaderboard, calibration
│   │   └── reliability/         # features, Model 2, thresholds, mitigation, evaluation
│   ├── storage/                 # per-session saved artifacts (git-ignored)
│   └── tests/
└── frontend/
    └── src/
        ├── pages/               # Upload, Training, Report, Predict, Monitoring
        ├── components/
        ├── api/                 # API client
        └── ...
```

---

## 11. Sessions and Background Training

- Every dataset upload creates a **session** with a unique `session_id`.
- Everything is saved under `backend/storage/{session_id}/`: raw CSV, splits info, fitted pipeline, Model 1, Model 2, thresholds, report JSON, status file.
- Models are saved with `joblib`.
- Training can take seconds to minutes, so:
  1. `POST /api/train/{id}` starts training as a **background task**.
  2. Frontend **polls** `GET /api/train/{id}/status` every 2 seconds.
  3. Status values: `pending`, `running` (with stage name), `completed`, `failed` (with error message).
  4. Stages: `profiling`, `splitting`, `model_selection`, `reliability_features`, `reliability_training`, `evaluation`, `saving`.
- Implement session cleanup for old sessions.

---

## 12. API Contract

Base path: `/api`. CORS enabled for `http://localhost:5173`. Swagger docs at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Returns `{"status": "ok"}` |
| POST | `/api/datasets/upload` | Upload CSV. Returns `session_id`, columns, preview rows |
| GET | `/api/datasets/{id}/profile` | Rows, types, missing values, class balance, detected task |
| POST | `/api/train/{id}` | Body: `{"target_column": "..."}`. Starts background training |
| GET | `/api/train/{id}/status` | `pending` / `running` (stage) / `completed` / `failed` |
| GET | `/api/report/{id}` | Leaderboard, selected model, metrics, chart data (ModelDoctor vs baseline) |
| GET | `/api/schema/{id}` | Feature names, types, allowed categories, ranges, so React can auto-build the input form |
| POST | `/api/predict/{id}` | Single record. Returns prediction, confidence, trustworthiness, decision, warnings |
| POST | `/api/predict/{id}/batch` | CSV upload. Returns a table of results |
| POST | `/api/feedback/{id}` | Actual outcomes for monitoring |
| GET | `/api/monitoring/{id}` | Coverage, decision distribution, observed vs estimated reliability |

**Predict request (example)**
```json
{
  "features": {
    "income": 42000,
    "age": 35,
    "loan_amount": 15000,
    "employment_type": "self-employed"
  }
}
```

**Predict response (example)**
```json
{
  "prediction": "No default",
  "confidence": 0.97,
  "trustworthiness": 0.68,
  "decision": "REVIEW",
  "warnings": [
    "income outside training range",
    "2 missing fields",
    "model disagreement high"
  ]
}
```

**Report response (shape)**
```json
{
  "task": "classification",
  "selected_model": "Random Forest",
  "leaderboard": [{"model": "Random Forest", "score": 0.77, "train_seconds": 3.2}],
  "overall_error_rate": 0.14,
  "comparison": [
    {"method": "No filtering", "coverage": 1.0, "selective_risk": 0.14},
    {"method": "Confidence baseline", "coverage": 0.70, "selective_risk": 0.09},
    {"method": "ModelDoctor", "coverage": 0.70, "selective_risk": 0.06}
  ],
  "auroc": {"confidence_baseline": 0.71, "modeldoctor": 0.82},
  "risk_coverage_curve": {"modeldoctor": [], "baseline": []},
  "calibration_curve": []
}
```

(The agent should write full Pydantic schemas and show them for approval **before** implementing endpoints. The same schemas are then given to the frontend.)

**Error handling:** return clear JSON errors with proper status codes (400 bad input, 404 unknown session, 409 training not finished, 422 validation, 500 internal). Never return stack traces to the client.

---

## 13. Frontend Pages

1. **Upload:** drag-and-drop CSV, preview table, target column selector.
2. **Training progress:** stage indicator driven by polling the status endpoint.
3. **System report:** dataset profile, model leaderboard, ModelDoctor vs baseline (risk-coverage chart, comparison table), calibration plot.
4. **Predict:** form auto-generated from `/api/schema`, plus batch CSV upload. Result card shows prediction, confidence, **trustworthiness %**, a colored **TRUST / REVIEW / REJECT** badge, and warnings.
5. **Monitoring:** decision distribution, coverage over time, feedback submission, observed vs estimated reliability.

UI requirements:
- Show **confidence and trustworthiness side by side** with a short explanation of the difference. This gap is the core idea of the project.
- Clear loading, error, and empty states.
- Responsive layout.

---

## 14. Build Order (one task at a time)

Complete and verify each step before starting the next. Each step includes a small test.

1. **Project setup:** folder structure, `requirements.txt`, FastAPI skeleton with CORS and `/api/health`, virtual environment, confirm server runs.
2. **Data layer:** CSV loading, profiling, task detection, preprocessing pipeline, three-way split.
3. **Model 1:** candidate pool, cross-validation, leaderboard, calibration, retrain winner.
4. **Reliability features:** model-agnostic feature extractor.
5. **Model 2 and thresholds:** correctness labels, reliability model, threshold selection, mitigation, warnings.
6. **Evaluation:** metrics vs confidence baseline, all returned as JSON.
7. **API layer:** write schemas first, then sessions, background training, status, schema, predict, batch, report.
8. **Backend test:** run the full flow via `/docs` on a real dataset from `sample_data/`.
9. **React app:** upload and progress first, then report, then predict, then monitoring.
10. **Integration and polish:** end-to-end testing, error handling, regression support, feedback and monitoring, README.

After steps 3 and 5, **review the code explicitly for data leakage.**

---

## 15. Testing Requirements

- Unit tests for each ML module (pytest).
- A **leakage test**: assert no index overlap between train, calibration, and test splits; assert Model 2's feature matrix does not contain the target.
- A test that Model 2 works regardless of which algorithm Model 1 is (loop over all candidates).
- API tests for each endpoint, including error cases.
- An end-to-end test on a small sample dataset.

---

## 16. Known Limitations (document honestly)

- Works on tabular data only.
- Small or very noisy datasets give Model 2 too little signal.
- Trustworthiness is an estimate learned from history ("among past predictions that looked like this, about X% were correct"), not a guarantee for one specific case.
- If future data drifts far from the training data, reliability estimates can degrade; monitoring and retraining address this.
- Rejecting too many predictions makes the system useless; there is always an error-vs-coverage trade-off.
- Decision Trees produce coarse probabilities; calibration is applied but may not fully fix this.

---

## 17. Definition of Done

- User can upload a CSV, select a target, and watch training progress.
- System automatically selects Model 1 and shows a leaderboard.
- Report shows ModelDoctor vs the confidence baseline with charts.
- User can submit a new record or batch and receive prediction, confidence, trustworthiness %, warnings, and TRUST / REVIEW / REJECT.
- Monitoring page works with submitted feedback.
- All tests pass; README explains setup and usage.
- Results demonstrated on at least 3 different datasets.

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

5. **Reject band on high-error models (end-to-end build):**
   - With a fixed 30% requirement, a multiclass model with 32% overall error sent 70% of calibration rows to REJECT (38.8% error, i.e. most of them correct) and left a single row in REVIEW. The requirement is now relative to the base error rate (see 7.5); on that dataset the REJECT band is correctly disabled and those predictions go to REVIEW.

6. **Test-split results on three datasets (`RESULTS.md`):**
   - Loans (binary), customer tiers (3 classes) and the real Wisconsin breast-cancer data: ModelDoctor matches the calibrated-confidence baseline at equal coverage (within about 0.2 points). On breast cancer, the calibration split has fewer than 30 errors, so the system correctly uses the confidence fallback.
   - With each algorithm forced as Model 1, differences range from -1.6 to +1.6 points of selective risk. There is no consistent gain on i.i.d. data. The benefit of the reliability layer is per-record: out-of-range, unseen-category, missing and far-from-training inputs get warnings and much lower trustworthiness than Model 1's confidence.

# Reviewer-Safe Workflow

This repo can be used in two different ways, and mixing them creates weak claims:

- `LIAR` as the primary claim-verification benchmark
- `ISOT` as an auxiliary article-classification benchmark

Do not present ISOT as the main external-evidence fact-verification result. In this codebase, ISOT is better treated as a strong article-classification benchmark with optional evidence augmentation.

## 1. Review-Safe Environment

Use these environment variables for reviewer-facing runs:

```powershell
$env:DATASET="liar"
$env:BENCHMARK_PROFILE="verification"
$env:CHEAP_MODE="false"
$env:STRICT_REVIEW_MODE="true"
$env:SAVE_ALL_EVAL_RESULTS="false"
```

For ISOT auxiliary runs:

```powershell
$env:DATASET="isot"
$env:BENCHMARK_PROFILE="article_classification"
$env:CHEAP_MODE="false"
```

## 2. Rebuild Data and Knowledge Base

Rebuild processed data after changing the dataset or splitting logic:

```powershell
python modules\preprocessor.py
```

Rebuild the knowledge base after changing datasets or provenance logic:

```powershell
python modules\knowldege_base_builder.py
```

Important:

- LIAR KB is the better fit for verification-style runs.
- ISOT KB built from Reuters train articles should be treated as auxiliary retrieval for article classification, not as independent external evidence.

## 3. Train Core Models

Train RoBERTa:

```powershell
python modules\bert_classifier.py
```

Train the stacking ensemble:

```powershell
python modules\ensemble.py
```

The ensemble script now reports both fit metrics and a small holdout estimate before the final refit.

## 4. Run Evaluation

Main evaluation:

```powershell
python evaluation\metrics.py
```

Ablation:

```powershell
python evaluation\ablation.py
```

Reviewer audit:

```powershell
python evaluation\reviewer_audit.py
```

## 5. How to Present Results

- Lead with LIAR for the main verification claim.
- Use ISOT as supporting evidence for article classification only.
- Report whether cheap mode was off.
- Report `coverage`, `n_requested`, and `n_failed` from the saved evaluation JSON.
- Treat `hallucination_rate` as `N/A` for configurations where hallucination detection is disabled.
- Do not claim that train-derived Reuters retrieval is equivalent to independent evidence retrieval.

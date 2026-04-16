# Reviewer Workflow

EvidenceChain supports two evaluation modes:

- `LIAR` as the primary claim-verification benchmark
- `ISOT` as an auxiliary article-classification benchmark

Do not present ISOT as the main external-evidence fact-verification result. In this repository it is better treated as article classification with optional evidence augmentation.

## Review-Safe Environment

Use this profile for reviewer-facing LIAR runs:

```powershell
$env:DATASET = "liar"
$env:BENCHMARK_PROFILE = "verification"
$env:CHEAP_MODE = "false"
$env:STRICT_REVIEW_MODE = "true"
$env:SAVE_ALL_EVAL_RESULTS = "false"
```

Use this profile for ISOT auxiliary runs:

```powershell
$env:DATASET = "isot"
$env:BENCHMARK_PROFILE = "article_classification"
$env:CHEAP_MODE = "false"
```

## Recommended Run Order

Rebuild processed splits:

```powershell
python -m src.preprocessing.preprocessor
```

Rebuild the knowledge base:

```powershell
python -m src.rag.knowledge_base_builder
```

Train the RoBERTa classifier:

```powershell
python -m src.models.bert_classifier
```

Train the stacking ensemble:

```powershell
python -m src.models.ensemble
```

Run the full evaluation:

```powershell
python -m src.evaluation.metrics
```

Run ablations:

```powershell
python -m src.evaluation.ablation
```

Generate the reviewer audit:

```powershell
python -m src.evaluation.reviewer_audit
```

Generate SHAP feature explanations:

```powershell
python -m src.evaluation.shap_explainer
```

## Reporting Guidance

- Lead with LIAR when making claim-verification statements.
- Use ISOT as supporting evidence for article classification only.
- Report whether `CHEAP_MODE` was disabled.
- Include `coverage`, `n_requested`, and `n_failed` from the saved evaluation JSON.
- Treat `hallucination_rate` as `N/A` for configurations where hallucination detection is disabled.
- Do not present train-derived Reuters retrieval as equivalent to fully independent external evidence.

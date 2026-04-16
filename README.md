# EvidenceChain

EvidenceChain is a research-oriented fake-news and claim-verification framework that combines claim decomposition, retrieval-augmented evidence reasoning, neural text classification, and ensemble learning in a single reproducible pipeline.

The project is designed around a simple idea: a model should not only predict whether a claim is fake or real, it should expose the evidence, reasoning path, and uncertainty signals behind that decision. EvidenceChain therefore pairs a RoBERTa classifier with a retrieval-and-reasoning branch and then fuses both views into a final verdict.

## Key Features

- Claim decomposition for breaking compound claims into atomic sub-claims
- RAG-based evidence retrieval with a FAISS-backed knowledge base
- Credibility and temporal scoring for weighted evidence ranking
- RoBERTa classifier for strong text-only baseline performance
- Ensemble learning via a logistic-regression stacking model
- Explainability with SHAP for feature-level interpretation of ensemble behavior

## System Architecture

The end-to-end pipeline follows this sequence:

1. Preprocess the input claim or article text.
2. Decompose complex claims into independently verifiable sub-claims.
3. Retrieve evidence from a dataset-specific knowledge base.
4. Re-rank evidence using semantic similarity, source credibility, and recency.
5. Reason over the retrieved evidence with an LLM and produce sub-claim verdicts.
6. Aggregate sub-claim verdicts into a RAG summary.
7. Score the original input with a fine-tuned RoBERTa classifier.
8. Fuse RAG and classifier signals with a stacking ensemble.
9. Evaluate the system, export reviewer-facing artifacts, and optionally generate SHAP explanations.

Additional implementation detail is documented in [architecture.md](docs/architecture.md) and [reviewer_workflow.md](docs/reviewer_workflow.md).

## Tech Stack

- Python
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- scikit-learn
- pandas and NumPy
- matplotlib and seaborn
- SHAP
- Optional LLM backends: Ollama, Groq, Gemini, and OpenAI-compatible APIs

## Dataset Information

EvidenceChain supports two datasets with different roles:

- LIAR: the primary benchmark for claim-level verification and reviewer-facing fact-checking claims
- ISOT: an auxiliary article-classification benchmark used to study longer-form inputs

Dataset positioning guidance is documented in [datasets.md](docs/datasets.md).

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set environment variables in a local `.env` file when using cloud LLM providers. The repository ignores `.env` by default.

Example configuration:

```env
DATASET=liar
BENCHMARK_PROFILE=verification
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
```

## Usage

### Preprocess datasets

```powershell
python -m src.preprocessing.preprocessor
```

### Build the knowledge base

```powershell
python -m src.rag.knowledge_base_builder
```

### Train the RoBERTa classifier

```powershell
python -m src.models.bert_classifier
```

### Train the stacking ensemble

```powershell
python -m src.models.ensemble
```

### Run the full evaluation

```powershell
python -m src.evaluation.metrics
```

### Run ablations

```powershell
python -m src.evaluation.ablation
```

### Generate the reviewer audit

```powershell
python -m src.evaluation.reviewer_audit
```

### Generate SHAP explanations

```powershell
python -m src.evaluation.shap_explainer
```

### Verify custom claims

```powershell
python -m src.pipeline --claim "COVID vaccines cause infertility and are banned in Europe"
```

## Project Structure

```text
Evidence-Chain/
├── config/
├── data/
├── docs/
├── results/
├── scripts/
├── src/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   └── rag/
├── LICENSE
├── README.md
└── requirements.txt
```

Directory overview:

- `config/`: shared configuration and path resolution
- `data/`: raw, processed, and local generated dataset assets
- `docs/`: architecture, dataset notes, reviewer workflow, and research draft material
- `results/`: exported tables, figures, and local model outputs
- `scripts/`: reproducible workflow helpers
- `src/preprocessing/`: dataset loading and normalization
- `src/models/`: RoBERTa training and ensemble fusion
- `src/rag/`: decomposition, retrieval, LLM reasoning, and KB building
- `src/evaluation/`: metrics, ablations, audits, and SHAP explainability

## Roadmap

- Expand the evidence base beyond the current prototype corpora
- Rebuild retrieval around chunk-level indexing instead of document-level summaries
- Add stronger benchmark reporting for cross-dataset generalization
- Improve reviewer-facing experiment tracking and reproducibility metadata
- Add automated tests for preprocessing, retrieval, and evaluation utilities
- Broaden explainability support beyond ensemble-level SHAP summaries

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

# System Architecture

EvidenceChain is organized as a modular verification pipeline with four main layers.

## 1. Preprocessing

Location: `src/preprocessing/`

- loads LIAR and ISOT
- normalizes claim or article text
- creates model-ready structured inputs
- writes dataset-specific processed splits

## 2. Retrieval-Augmented Verification

Location: `src/rag/`

- decomposes complex claims into atomic sub-claims
- retrieves evidence from a FAISS index
- applies source-credibility and temporal weighting
- reasons over retrieved evidence with an LLM
- flags low-grounding, high-confidence cases as potential hallucinations

## 3. Modeling and Fusion

Location: `src/models/`

- fine-tunes `roberta-base` for binary fake/real classification
- learns a logistic-regression stacking ensemble over classifier and RAG signals
- exports feature tables used for ensemble inspection and SHAP analysis

## 4. Evaluation and Reporting

Location: `src/evaluation/`

- runs full evaluation and ablations
- saves reviewer-facing metrics and audit summaries
- produces confusion matrices and confidence plots
- explains ensemble feature importance with SHAP

## Pipeline Flow

1. Load a claim or article example.
2. Clean and normalize the text.
3. Decompose the input into independently verifiable sub-claims.
4. Retrieve weighted evidence for each sub-claim.
5. Produce a grounded verdict for each sub-claim with confidence and rationale.
6. Aggregate sub-claim verdicts into a RAG summary.
7. Score the original input with the RoBERTa classifier.
8. Fuse neural and retrieval signals with the stacking ensemble.
9. Save metrics, audit artifacts, and optional explainability outputs.

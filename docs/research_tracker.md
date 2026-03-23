# EvidenceChain Research Tracker

## Purpose

This file is the single source of truth for:

- what the project is using
- why it is using it
- what is already implemented
- what is still pending

Update this file whenever the code, results, or paper changes.

## Current Scope

- Current reproducible system: LIAR-based EvidenceChain pipeline plus
  LIAR and ISOT RoBERTa baselines
- ISOT preprocessing: complete
- ISOT RoBERTa training: complete
- Current paper state: draft updated with LIAR and ISOT baseline results

## Core Choices

| Area | Current choice | Why this choice | Current status |
| --- | --- | --- | --- |
| Main short-claim dataset | LIAR | Short political claims plus metadata make it useful for decomposition and claim verification | Complete |
| Main long-article dataset | ISOT | Long articles test article-level language modeling and generalization | Baseline complete |
| Text classifier | `roberta-base` | Strong transformer baseline for binary classification | LIAR and ISOT checkpoints saved |
| Embedding model | `all-MiniLM-L6-v2` | Fast sentence embeddings for semantic retrieval | In use |
| Vector store | FAISS `IndexFlatIP` | Simple and reliable cosine-similarity retrieval | In use |
| Evidence sources | WHO, Reuters, Wikipedia | Mix of credibility and coverage for a prototype knowledge base | Current KB has 50 docs |
| LLM provider | Groq | Fast hosted inference for decomposition and reasoning | In use |
| LLM model | `llama-3.3-70b-versatile` | Strong general reasoning model with deterministic temperature 0.0 | In use |
| Hallucination rule | similarity < 0.45 and confidence > 0.75 | Simple groundedness diagnostic | In use |
| Final combiner | Logistic Regression ensemble | Interpretable meta-classifier over a small feature set | Needs real validation-set retraining |

## Implemented Modules

| Module | File | What it does | Status |
| --- | --- | --- | --- |
| Data loading | `modules/data_loader.py` | Loads LIAR and ISOT from local files | Complete |
| Preprocessing | `modules/preprocessor.py` | Cleans text and builds model inputs | Complete |
| RoBERTa baseline | `modules/bert_classifier.py` | Trains and evaluates the text-only baseline | LIAR and ISOT complete |
| Knowledge base builder | `modules/knowldege_base_builder.py` | Builds the FAISS index and metadata | Prototype complete |
| Retriever | `modules/retriever.py` | Retrieves weighted evidence | Complete |
| Decomposer | `modules/decomposer.py` | Breaks claims into atomic sub-claims | Complete |
| Reasoner | `modules/reasoner.py` | Produces verdict, confidence, and explanation from evidence | Complete |
| Ensemble | `modules/ensemble.py` | Combines RoBERTa and RAG features | Logic complete, training workflow still provisional |
| Full pipeline | `pipeline.py` | Connects all modules end to end | Complete |
| Evaluation | `evaluation/metrics.py` | Computes metrics and plots | Complete, rerun needed after latest fixes |
| Ablation | `evaluation/ablation.py` | Removes components and measures impact | Complete, rerun needed after latest fixes |

## Current Result Artifacts

| Artifact | Meaning | Current value |
| --- | --- | --- |
| `results/tables/bert_results.csv` | Saved LIAR RoBERTa baseline | Accuracy 0.6461, weighted F1 0.6443 |
| `results/tables/bert_results_isot.csv` | Saved ISOT RoBERTa baseline | Accuracy 0.9996, weighted F1 0.9996 |
| `results/tables/full_evaluation.json` | Saved 50-claim LIAR pipeline sample | Accuracy 0.5000, weighted F1 0.4833, ROC-AUC 0.6208, HR 0.46 |
| `results/tables/ablation_results.csv` | Saved 30-claim LIAR ablation sample | BERT-only currently beats full pipeline |
| `results/pipeline_test.json` | Qualitative sample outputs | 3 handpicked claims |

## Important Notes

1. The current knowledge base is still small. It contains 50 documents.
2. The saved index is document-level, not chunk-level.
3. The ensemble still needs training on real validation-set features.
4. The ISOT score is extremely high and should be interpreted with care,
   because the dataset is likely easier and more source-biased than
   LIAR.

## Next Actions

1. Train the ensemble on real validation features.
2. Add ISOT and cross-dataset tables to `docs/paper.md`.
3. Rerun full evaluation and ablation with the corrected code.
4. Add SHAP analysis and human evaluation outputs.
5. Run cross-dataset experiments to test generalization.

## Documentation Rules

Whenever you change the project:

1. Update `docs/journal.md` with what changed and what is still pending.
2. Update `docs/paper.md` if the methodology or results changed.
3. Update `docs/simple_notes.md` so the plain-English explanation stays
   aligned with the real system.
4. Update this tracker if any model, dataset, metric, or tool choice
   changes.

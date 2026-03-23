# EvidenceChain Journal

## 2026-03-23 - Documentation and Reporting Pass

### What was completed in this pass

- Rewrote `docs/paper.md` so it reflects the real project state instead
  of placeholders and optimistic future claims.
- Added `docs/research_tracker.md` to track what is being used, why it
  is being used, and what is still pending.
- Rewrote `docs/simple_notes.md` into a cleaner plain-English version of
  the project.
- Fixed the ablation path so "no weighting" now removes weighting during
  retrieval instead of only after ranking.
- Tightened evaluation reporting so evidence recall depends on retrieval
  similarity instead of counting every claim as relevant.
- Expanded pipeline outputs so sub-claim level analysis can be tracked
  more honestly in future evaluation runs.

### Current reproducible state

- LIAR preprocessing: complete
- ISOT preprocessing: complete
- LIAR RoBERTa baseline: complete
- ISOT RoBERTa baseline: complete
- FAISS knowledge base: complete as a prototype
- Claim decomposition: complete
- LLM reasoning: complete
- Hallucination flagging: complete
- Full LIAR pipeline: complete
- ISOT RoBERTa training: complete on Colab T4

### Current saved metrics

- LIAR RoBERTa baseline:
  - Accuracy: 0.6461
  - Weighted F1: 0.6443
  - Best epoch: 2

- ISOT RoBERTa baseline:
  - Accuracy: 0.999554565701559
  - Weighted F1: 0.999554565701559
  - Max length: 256
  - Best checkpoint: epoch 1

- Saved LIAR full-pipeline sample (50 claims):
  - Accuracy: 0.5000
  - Weighted F1: 0.4833
  - ROC-AUC: 0.6208
  - Hallucination rate: 0.46

- Saved LIAR ablation sample (30 claims):
  - Full EvidenceChain: 0.5333 accuracy, 0.5354 weighted F1
  - BERT-only: 0.6333 accuracy, 0.6243 weighted F1
  - RAG-only: 0.4333 accuracy, 0.2620 weighted F1

### What these results mean

The codebase is no longer just a folder of disconnected modules. It is a
working research prototype with a saved LIAR baseline, a saved knowledge
base, a full pipeline, and reporting artifacts.

At the same time, the current retrieval-plus-reasoning branch is not yet
strong enough to beat the RoBERTa-only baseline. That is the main
research bottleneck right now.

The new ISOT baseline is extremely high. That is useful as a result, but
it should be reported carefully because ISOT is likely much easier than
LIAR under the current split and may contain strong source/style cues.

### Important current limitations

1. The current knowledge base is small and document-level.
2. The ensemble still needs training on real validation-set features.
3. Cross-dataset evaluation is still missing.
4. Evaluation and ablation should be rerun after the latest reporting
   fixes.

### Next actions

1. Train the ensemble on real validation-set features.
2. Rerun full evaluation on LIAR with the corrected reporting code.
3. Rerun ablation after ensemble retraining.
4. Run cross-dataset evaluation.
5. Add SHAP analysis and human evaluation.

## Milestone Summary So Far

### Infrastructure

- Project structure created
- Virtual environment and dependencies prepared
- Central configuration created
- API key handling set up through `.env`

### Data

- LIAR raw data loaded and processed
- ISOT raw data loaded and processed

### Models

- LIAR RoBERTa model trained and saved
- ISOT RoBERTa model trained and saved
- Decomposition and reasoning modules connected

### Retrieval

- Knowledge base built and saved
- FAISS retrieval working
- Source and temporal weighting implemented

### Writing

- Paper draft upgraded from placeholder-heavy to a real working draft
- Journal cleaned and converted into a usable status log
- Simple notes aligned with the real project state
- Research tracker added for tool and design decisions

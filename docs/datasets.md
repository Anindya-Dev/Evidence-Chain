# Dataset Positioning

EvidenceChain uses two datasets, but they serve different research purposes and should not be presented interchangeably.

## LIAR

LIAR is the primary benchmark for claim-level verification in this repository.

Why it matters:

- the inputs are short human claims rather than full articles
- compound statements make claim decomposition meaningful
- the benchmark is better aligned with evidence retrieval and reasoning
- it is the correct benchmark for reviewer-facing verification claims

In practice, LIAR is the dataset to use when evaluating the full claim-verification pipeline.

## ISOT

ISOT is used as an auxiliary article-classification benchmark.

Why it is still useful:

- it tests longer-form text inputs
- it provides a stronger article-classification setting for the RoBERTa branch
- it helps study how retrieval behaves on article-style inputs

Important limitation:

- ISOT should not be treated as the main external-evidence fact-verification benchmark
- its labels are heavily entangled with source and style cues

In this repository, ISOT is best interpreted as article classification with optional evidence augmentation.

## Recommended Usage

- Use `DATASET=liar` and `BENCHMARK_PROFILE=verification` for reviewer-facing fact-verification runs.
- Use `DATASET=isot` and `BENCHMARK_PROFILE=article_classification` for supporting article-classification analysis.
- Keep dataset-specific knowledge bases in `data/knowledge_base/<dataset>/`.
- Keep dataset-specific evaluation artifacts in `results/tables/`.

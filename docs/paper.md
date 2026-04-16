# EvidenceChain: A Decomposition-Driven Retrieval-Augmented Framework for Explainable Fake News Detection

## Abstract

Fake news detection systems often rely on text-only classification,
which makes them fast but difficult to justify. They may classify a
claim as true or false without showing which evidence was used, they
usually treat compound claims as single units, and they rarely measure
whether a generated explanation is actually grounded in retrieved
evidence. This project proposes EvidenceChain, a modular fact-checking
pipeline that combines claim decomposition, weighted evidence retrieval,
LLM-based evidence reasoning, a RoBERTa classifier, and a stacking
ensemble meta-classifier. The system introduces three main ideas:
(1) decomposition of compound claims into independently verifiable
sub-claims, (2) source-credibility and temporal weighting during
retrieval, and (3) an explicit hallucination flag based on the
relationship between retrieval similarity and LLM confidence. The
current repository contains a completed LIAR-based pipeline, processed
LIAR and ISOT datasets, saved LIAR and ISOT RoBERTa checkpoints, a
saved knowledge base, and evaluation and ablation scripts. On LIAR, the
RoBERTa baseline currently achieves 0.6461 accuracy and 0.6443 weighted
F1. On ISOT, a RoBERTa baseline with maximum sequence length 256
achieves 0.9996 accuracy and 0.9996 weighted F1. The current end-to-end
LIAR pipeline, evaluated on a saved sample of 50 claims, achieves
0.5000 accuracy, 0.4833 weighted F1, 0.6208 ROC-AUC, and a
hallucination rate of 0.46. These results show that the retrieval and
reasoning stack is implemented, but the current combined LIAR system
still requires calibration before it can reliably outperform the
text-only baseline.

## 1. Introduction

The spread of misinformation across news websites, social media, and
political communication has made automatic fake news detection an
important research problem. A useful fake news system must do more than
output a label. It should explain why a claim is judged to be true or
false, identify the evidence used in that decision, and expose cases
where the model is uncertain or unsupported.

Most existing fake news classifiers operate directly on text and learn
stylistic patterns that correlate with misinformation. These methods can
be strong baselines, especially on article-level datasets such as ISOT,
but they have two practical weaknesses. First, they do not explicitly
reason over external evidence. Second, they struggle to explain how a
decision was made. A claim may be labeled false because of linguistic
patterns, even when the model has not actually verified the claim.

Retrieval-based verification methods improve interpretability by
bringing external evidence into the decision loop. However, they also
have important limitations in practice. Compound claims are often passed
to retrieval as single units, even when different parts of the claim
require different evidence. Retrieved documents are commonly ranked only
by semantic similarity, which ignores both source credibility and
recency. In addition, LLM-based reasoning layers can produce confident
answers that sound grounded while actually relying on parametric memory
instead of the retrieved context.

EvidenceChain is designed to address these gaps through a modular
research pipeline. The system first preprocesses and normalizes the
claim, then decomposes it into atomic sub-claims, retrieves evidence
from a FAISS index, reasons over that evidence with an LLM, scores
possible hallucination, adds a RoBERTa-based text signal, and finally
combines the symbolic and neural signals in a stacking ensemble.

The main contributions of this project are:

1. A decomposition-first verification pipeline for compound claims.
2. A retrieval stage that incorporates source credibility and temporal
   relevance in addition to semantic similarity.
3. A simple, explicit hallucination criterion for identifying LLM
   verdicts that appear overly confident relative to the retrieved
   evidence.

This paper documents the current project state. The LIAR pipeline is
implemented and evaluated. The paper is therefore written as a living
draft: LIAR and ISOT baseline results are now concrete, while the
cross-dataset and final full-pipeline experiments remain the next
update.

## 2. Related Work

Research on fake news detection has largely followed two directions:
text-only classification and evidence-based verification.

Text-only classification approaches learn patterns from the claim text or
article text itself. The LIAR benchmark popularized this setting by
providing short political claims with metadata such as speaker,
affiliation, and context. Transformer models such as BERT and RoBERTa
have since become standard baselines because they capture contextual
semantics more effectively than earlier bag-of-words or recurrent
architectures. These models are especially strong when the task is
driven by writing style, framing, or speaker cues.

Evidence-based verification systems instead retrieve external documents
and reason over them. FEVER-style pipelines and later retrieval-augmented
architectures showed that evidence improves interpretability and makes it
possible to justify a verdict rather than only predict it. In recent
RAG-based systems, dense retrievers and LLMs are often combined to
select relevant evidence and then generate a verdict or explanation from
that evidence.

Despite these advances, three gaps remain relevant to this project.
First, many verification pipelines still assume that an input claim is
already atomic. This assumption breaks when a statement contains several
independently verifiable parts. Second, retrieval systems often rank
documents only by semantic similarity, without explicitly rewarding more
credible sources or more recent information. Third, hallucination in the
reasoning layer is usually discussed qualitatively rather than measured
through a transparent and reproducible rule.

EvidenceChain fits at the intersection of these areas. It retains a
strong text-only baseline through RoBERTa, adds explicit retrieval and
reasoning, and introduces decomposition and hallucination tracking as
first-class components.

## 3. Methodology

### 3.1 System Overview

EvidenceChain is a modular pipeline with the following stages:

1. Text preprocessing and normalization.
2. Claim decomposition into atomic sub-claims.
3. Evidence retrieval from a FAISS knowledge base.
4. LLM reasoning over retrieved evidence.
5. Hallucination detection based on similarity and confidence.
6. RoBERTa classification of the original claim text.
7. Stacking ensemble combination of neural and retrieval signals.

The main implementation entry point is `src/pipeline.py`, which loads the
decomposer, retriever, reasoner, and ensemble once and then evaluates
claims one at a time.

### 3.2 Data Preprocessing

The preprocessing module supports both LIAR and ISOT. For LIAR, the
system lowercases text, removes URLs, strips non-alphanumeric symbols
except apostrophes, removes the leading "says" artifact when present,
and normalizes whitespace. Missing metadata fields such as `job_title`,
`state_info`, and `context` are filled with the token `unknown`. The
claim is then combined with selected metadata into a structured input:

`[CLAIM] <claim> [SPEAKER] <speaker> [SUBJECT] <subject>`

This design preserves contextual information that may help the RoBERTa
classifier distinguish subtle political claims.

For ISOT, the system uses title plus truncated body text. The title is
retained because fake and real news often differ strongly in headline
style. The article body is limited to the first 400 words so it fits
within RoBERTa's token budget.

### 3.3 Claim Decomposition

EvidenceChain treats claim decomposition as a core step rather than a
post-processing trick. Many real-world statements contain multiple
assertions. For example, the sentence "COVID vaccines cause infertility
and are banned in Europe" contains at least two independent factual
sub-claims. Verifying such a sentence as a single unit can mix evidence
from separate topics and reduce reliability.

The decomposition module uses an LLM prompt that asks for a JSON array
of atomic, independently verifiable sub-claims. The prompt explicitly
forbids adding new information and falls back to the original claim if
JSON parsing fails. Temperature is fixed at zero for deterministic
behavior.

### 3.4 Evidence Base and Retrieval

The retrieval layer is implemented with Sentence-Transformers and FAISS.
Documents are embedded using `all-MiniLM-L6-v2`, normalized, and stored
in an `IndexFlatIP` index so cosine similarity search can be performed
efficiently.

The current saved knowledge base contains 50 documents drawn from
trusted hard-coded summaries plus Wikipedia pages. Source distribution in
the saved metadata is:

- WHO: 2
- Reuters: 16
- Wikipedia: 32

Each document also stores a source credibility weight and a recency
weight. During standard retrieval, the ranking score is:

`ranking_score = similarity * source_weight + recency_weight`

This design aims to reduce the effect of high-similarity but weak or
stale evidence. A no-weighting mode is also available for ablation.

One important implementation note is that the current saved index is
document-level rather than chunk-level. Chunk size and overlap are
already defined in configuration, but the present knowledge base stores
one embedding per document. This is a known limitation and an obvious
next improvement.

### 3.5 LLM Reasoning and Hallucination Detection

For each sub-claim, the reasoner builds a structured prompt containing
the sub-claim and the retrieved evidence. The LLM is instructed to use
only the given evidence and to return a JSON object with a verdict,
confidence score, and one-sentence explanation.

The reasoner supports three verdicts:

- `TRUE`
- `FALSE`
- `UNVERIFIABLE`

The module then applies a hallucination rule:

`hallucinated = (max_similarity < 0.45) and (confidence > 0.75)`

This rule is intentionally simple. It does not claim to solve the full
groundedness problem, but it gives the system an explicit diagnostic for
cases where the LLM sounds more certain than the retrieved evidence
justifies.

### 3.6 RoBERTa Baseline

The text-only baseline is a fine-tuned `roberta-base` binary classifier.
For LIAR, the original six labels are mapped to two classes:

- REAL: `true`, `mostly-true`, `half-true`
- FAKE: `barely-true`, `false`, `pants-fire`

The model is trained for three epochs with batch size 16 and learning
rate `2e-5`. The best checkpoint is selected by validation F1. This
baseline is important because it measures how far the project can get
using claim text alone, without retrieval or reasoning.

### 3.7 Stacking Ensemble

The final prediction layer is a logistic regression meta-classifier that
combines six features:

1. RoBERTa probability of the REAL class.
2. RAG verdict confidence.
3. Maximum retrieved evidence similarity.
4. Average source credibility weight.
5. Hallucination flag.
6. Number of sub-claims.

The intended design is to train the ensemble on validation-set features
to avoid leakage. The repository already contains the ensemble class and
a saved ensemble artifact, but real validation-set calibration remains an
important project task before final claims about full-system superiority
should be made.

## 4. Experimental Setup

### 4.1 Datasets

This project uses two benchmark datasets.

**LIAR** is a benchmark of short political claims with speaker metadata.
It is well suited to claim-level reasoning and decomposition because the
claims are concise and often rhetorically dense.

**ISOT Fake News Dataset** contains full news articles labeled as real or
fake. It is useful for testing article-level language modeling and
generalization to longer inputs.

At the time of this draft:

- LIAR preprocessing, training, and baseline evaluation are complete.
- ISOT preprocessing, training, and baseline evaluation are complete.
- Cross-dataset evaluation and final ensemble recalibration are still
  pending.

### 4.2 Configuration

The main experimental settings in the repository are:

- Random seed: 42
- RoBERTa model: `roberta-base`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval depth: top 5
- Hallucination thresholds: similarity 0.45, confidence 0.75
- LLM provider: Groq
- LLM model: `llama-3.3-70b-versatile`

### 4.3 Implemented Comparisons

The repository currently supports the following implemented comparisons:

- RoBERTa-only baseline
- RAG-only pipeline
- Full EvidenceChain pipeline
- Component-removal ablations

Additional comparisons, especially cross-dataset experiments and a
classical TF-IDF plus logistic regression baseline, remain good next
steps once the ISOT checkpoint is available.

### 4.4 Evaluation Metrics

The current evaluation module computes:

- Accuracy
- Precision
- Recall
- Weighted F1
- ROC-AUC
- Hallucination Rate
- Evidence Recall

Confusion matrix and confidence distribution plots are also supported.
Because the evidence recall logic was tightened in this pass, that metric
should be rerun before being treated as final in the paper.

## 5. Current Results

### 5.1 RoBERTa Baselines on LIAR and ISOT

The saved RoBERTa baseline results are:

| Model | Dataset | Max Length | Accuracy | Weighted F1 | Epochs | Best Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RoBERTa-base | LIAR | 512 | 0.6461 | 0.6443 | 3 | 2 |
| RoBERTa-base | ISOT | 256 | 0.9996 | 0.9996 | 3 | 1 |

These numbers show that a text-only transformer baseline is already
reasonably competitive on LIAR and almost saturated on ISOT. This
contrast is important. LIAR behaves like a harder claim-verification
benchmark, while ISOT appears to be much easier under the current split.
The ISOT result should therefore be interpreted carefully, because it
likely reflects strong source and style cues in addition to genuine
semantic understanding.

### 5.2 Current End-to-End LIAR Pipeline

The saved full-pipeline evaluation currently covers a 50-claim LIAR
sample. The results are:

| System | Sample Size | Accuracy | Weighted F1 | ROC-AUC | Hallucination Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| EvidenceChain (current sample run) | 50 | 0.5000 | 0.4833 | 0.6208 | 0.4600 |

At this stage, the integrated system does not yet outperform the
RoBERTa-only baseline. This matters for interpretation: the project has
successfully implemented the retrieval, decomposition, and reasoning
modules, but the current retrieval corpus, aggregation strategy, and
ensemble calibration are not yet strong enough to consistently convert
those components into higher end-to-end accuracy.

### 5.3 Qualitative Behavior

The saved pipeline examples still demonstrate why the architecture is
worth pursuing.

1. On the vaccine claim, decomposition separates "causes infertility"
   from "banned in Europe", and both are contradicted by strong
   evidence. This is an ideal use case for the retrieval and reasoning
   stack.
2. On the unemployment-rate claim, the retrieval layer lacks sufficient
   temporal coverage, so the RAG branch returns `UNVERIFIABLE`. This
   exposes a real limitation of the current knowledge base rather than
   silently pretending certainty.
3. On the CDC self-destruct claim, the decomposition and reasoning
   pipeline correctly isolates a false operational claim and produces a
   FAKE verdict.

These examples suggest that the architecture is conceptually sound, but
the evidence base still needs broader topical coverage and stronger
calibration.

## 6. Preliminary Ablation Findings

The repository includes a saved preliminary ablation run over 30 LIAR
claims. The most important pattern is that the current full system is
not yet stronger than the RoBERTa-only baseline:

- Full EvidenceChain: 0.5333 accuracy, 0.5354 weighted F1
- BERT Only: 0.6333 accuracy, 0.6243 weighted F1
- RAG Only: 0.4333 accuracy, 0.2620 weighted F1

This diagnostic is useful. It suggests that the main current bottleneck
is not the text-only baseline but the retrieval-and-reasoning stack and
its fusion with the baseline. The saved ablation script has also now
been corrected so that the no-weighting setting removes weighting at the
retrieval stage rather than only after ranking. The ablation should
therefore be rerun after the ensemble is retrained on real validation
features.

## 7. Human Evaluation Protocol

Although human evaluation has not yet been executed, the protocol can be
defined clearly now.

### 7.1 Goals

The human study should answer four questions:

1. Does decomposition produce sensible atomic sub-claims?
2. Is the retrieved evidence relevant to the claim?
3. Is the final explanation faithful to the evidence?
4. Is the overall system output useful to a reader?

### 7.2 Proposed Setup

- Sample 100 claims from LIAR, with balanced REAL and FAKE labels.
- Ask 3 annotators to independently review system outputs.
- Provide the original claim, sub-claims, retrieved evidence, final
  verdict, and explanation.
- Score each item on a 1-5 scale for decomposition quality, evidence
  relevance, explanation faithfulness, and usefulness.

### 7.3 Agreement and Reporting

- Report mean score and standard deviation for each dimension.
- Compute inter-annotator agreement where applicable.
- Include at least 5 failure cases with short qualitative commentary.

This protocol is feasible with the current outputs and can be executed
once the LIAR and ISOT runs are frozen.

## 8. Conclusion

EvidenceChain is currently a functioning research prototype with a
completed LIAR preprocessing and baseline-training pipeline, a saved
knowledge base, claim decomposition, LLM reasoning, hallucination
tracking, and preliminary evaluation and ablation tooling. The project
already demonstrates the practical value of decomposition and explicit
evidence reasoning on selected examples. At the same time, the current
quantitative results show that the integrated system is not yet better
than the RoBERTa baseline, which is an important and honest outcome.

The next steps are clear:

1. Retrain the ensemble on real validation-set features.
2. Expand the evidence base beyond the current 50-document prototype.
3. Run cross-dataset experiments to test whether the very high ISOT
   score reflects genuine generalization or dataset bias.
4. Rerun full evaluation and ablation with the corrected reporting code.
5. Add SHAP-based feature analysis and the planned human evaluation.

With these updates, the project can move from a strong prototype into a
more complete research contribution.

## References

[Ahmed et al., 2020] Ahmed, H., Traore, I., and Saad, S. Detecting
opinion spams and fake news using text classification.

[Lewis et al., 2020] Lewis, P., Perez, E., Piktus, A., et al.
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.

[Liu et al., 2019] Liu, Y., Ott, M., Goyal, N., et al. RoBERTa: A
Robustly Optimized BERT Pretraining Approach.

[Reimers and Gurevych, 2019] Reimers, N. and Gurevych, I.
Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

[Thorne et al., 2018] Thorne, J., Vlachos, A., Christodoulopoulos, C.,
and Mittal, A. FEVER: a Large-scale Dataset for Fact Extraction and
VERification.

[Wang, 2017] Wang, W. Y. "Liar, Liar Pants on Fire": A New Benchmark
Dataset for Fake News Detection.

To be added soon

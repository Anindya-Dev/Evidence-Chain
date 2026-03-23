# EvidenceChain: A Decomposition-Driven Retrieval-Augmented 
# Generation Framework for Explainable Fake News Detection 
# with Temporal and Credibility-Aware Evidence Scoring

---

## Abstract
*(to be completed on Day 21)*

---

## 1. Introduction

The rapid proliferation of misinformation across digital platforms
poses a significant threat to public discourse, democratic processes,
and health outcomes. Despite growing research in automated fake news
detection, existing systems suffer from three fundamental limitations.

First, compound claims — statements containing multiple independently
verifiable assertions — are treated as single atomic units. This leads
to systematic false negatives when one sub-claim is true and another
is false, as the system cannot isolate the deceptive component.

Second, retrieval-based systems assign equal credibility to all
retrieved evidence regardless of source reliability or temporal
relevance. This is particularly problematic for evolving topics
such as policy decisions or medical research, where outdated
evidence may directly contradict current facts.

Third, large language model (LLM) outputs in verification pipelines
are rarely evaluated for groundedness — the degree to which a verdict
is actually supported by retrieved evidence rather than the model's
parametric memory. This leaves hallucination as an unmeasured
and uncontrolled source of error.

To address these limitations, we propose EvidenceChain, a
retrieval-augmented generation framework for fake news detection
that introduces three novel contributions:

C1 — Atomic claim decomposition prior to evidence retrieval,
     enabling independent verification of each sub-claim.

C2 — Temporally-weighted, source-credibility-aware evidence
     scoring that prioritizes recent, high-reliability documents.

C3 — A formal Hallucination Rate (HR) metric that measures the
     proportion of LLM verdicts unsupported by retrieved evidence,
     providing a reproducible benchmark for future work.

EvidenceChain combines these contributions with a fine-tuned
RoBERTa classifier and a stacking ensemble meta-classifier,
evaluated on the LIAR and ISOT benchmark datasets.

The remainder of this paper is organized as follows.
Section 2 reviews related work. Section 3 describes our
methodology. Section 4 presents experimental setup.
Section 5 reports results. Section 6 presents ablation study.
Section 7 describes human evaluation. Section 8 concludes.

---

## 2. Related Work
*(to be completed on Day 2)*

---

## 3. Methodology
*(built progressively from Day 2 onwards)*

### 3.1 System Overview

EvidenceChain processes an input claim through seven sequential
modules: (1) preprocessing, (2) claim decomposition, (3) weighted
evidence retrieval, (4) LLM reasoning with hallucination detection,
(5) BERT-based linguistic classification, (6) stacking ensemble,
and (7) explainability generation. Figure 1 illustrates the
complete pipeline architecture.

### 3.2 Experimental Configuration

All experiments use a fixed random seed of 42 to ensure full
reproducibility across dataset splits, model initialization,
and sampling operations.

**Model Selection:** The RoBERTa-base model [Liu et al., 2019]
was selected over BERT-base-uncased due to its superior
performance on text classification benchmarks. RoBERTa removes
the Next Sentence Prediction objective and trains on
significantly more data with larger batch sizes, consistently
yielding 2–4% accuracy improvements on classification tasks.

**Embedding Model:** Sentence embeddings are generated using
all-MiniLM-L6-v2, a 6-layer transformer producing 384-dimensional
vectors optimized specifically for semantic similarity tasks.
This model offers strong performance with fast inference,
making it suitable for large-scale evidence retrieval.

**Evidence Retrieval:** The top-5 most semantically similar
evidence chunks are retrieved per sub-claim (k=5), determined
empirically via validation set experiments. Document chunks
of 300 words with 50-word overlap are used to preserve
contextual continuity across chunk boundaries.

**Hallucination Detection:** Two thresholds govern hallucination
flagging: a minimum evidence similarity of τ_sim = 0.45 and
a maximum LLM confidence of τ_conf = 0.75. A verdict is
flagged as potentially hallucinated when evidence similarity
falls below τ_sim while LLM confidence exceeds τ_conf.
Both thresholds were selected via grid search on the
validation set and held fixed for all test evaluations.

### 3.3 Data Preprocessing

Raw claims from the LIAR dataset undergo a standardized 
preprocessing pipeline before being passed to any model component.

Text normalization includes lowercasing, URL removal, and 
elimination of non-alphanumeric characters with the exception 
of apostrophes, which are retained to preserve negation semantics 
(e.g., "is not" vs "isn't"). Numeric values are preserved as 
they carry factual significance in political claims. Claims 
beginning with the prefix "Says" — an artifact of PolitiFact's 
writing style — are stripped to isolate the core assertion.

Missing metadata fields (job_title: 28%, state_info: 21%, 
context: 1%) are imputed with the placeholder value "unknown" 
rather than empty strings, providing an explicit signal of 
metadata unavailability. Stopwords are intentionally retained, 
as RoBERTa processes full contextual sequences where negation 
words such as "not" carry critical semantic weight.

Each preprocessed instance combines the claim text with speaker 
identity and subject metadata into a structured input string:
[CLAIM] {text} [SPEAKER] {speaker} [SUBJECT] {subject}

This enriched representation allows RoBERTa to attend to 
speaker context alongside claim content, leveraging the 
metadata-aware architecture proposed in the original LIAR 
paper [Wang, 2017].

### 3.3 Claim Decomposition Module

Compound claims — statements containing multiple independently
verifiable assertions — present a fundamental challenge for
retrieval-based fact verification. When treated as a single
unit, mixed evidence from distinct sub-claims produces
unreliable aggregate verdicts.

EvidenceChain addresses this through an LLM-based claim
decomposition module that breaks each input claim into
atomic, independently verifiable sub-claims prior to
retrieval. The module prompts the LLM with explicit
constraints: each sub-claim must be a single verifiable
fact, no information may be added beyond what is present
in the original claim, and output must be structured as
a JSON array for reliable parsing.

Rule-based decomposition approaches — such as splitting
on conjunctions like "and" or "but" — fail on implicit
compound claims. For example, the claim "the CDC offices
will self-destruct in an emergency" contains no explicit
conjunction yet encodes two verifiable assertions: that
CDC offices exist in Atlanta (true), and that they are
designed to self-destruct (false). Our LLM-based approach
correctly identifies both. Temperature is fixed at 0.0
to ensure deterministic, reproducible decomposition.

This module constitutes Novel Contribution C1 of
EvidenceChain. Decomposition quality is evaluated
separately through manual annotation of 100 sampled
claims, reported in Section 7.
```

### 3.5 Evidence Retrieval Module

EvidenceChain employs a FAISS-based retrieval module to 
identify semantically relevant evidence for each sub-claim. 
Documents are encoded using the all-MiniLM-L6-v2 
sentence transformer, producing 384-dimensional embeddings 
that capture semantic meaning beyond keyword overlap.

The retrieval index uses IndexFlatIP with L2-normalized 
embeddings, enabling exact cosine similarity search. For 
each sub-claim, the top-k=5 most similar document chunks 
are retrieved and scored using our novel weighted formula:

weighted_score = similarity × source_weight + recency_weight

where source_weight reflects the credibility tier of the 
source (WHO=0.95, Reuters=0.90, Wikipedia=0.70, unknown=0.30) 
and recency_weight applies a linear decay bonus favouring 
documents published within a two-year window. This weighted 
scoring constitutes our second novel contribution (C2), 
ensuring that high-similarity evidence from unreliable or 
outdated sources does not dominate the verdict.
```

### 3.5 LLM Reasoning and Hallucination Detection

For each sub-claim, EvidenceChain constructs a structured 
prompt containing the sub-claim and its top-k retrieved 
evidence chunks, annotated with source credibility scores. 
The LLM is instructed to reason exclusively over provided 
evidence, explicitly prohibited from drawing on parametric 
memory. Output is constrained to a structured JSON object 
containing a verdict (TRUE, FALSE, or UNVERIFIABLE), a 
confidence score, and a one-sentence reasoning chain.

Temperature is fixed at 0.0 to ensure deterministic, 
reproducible verdicts across identical inputs.

**Hallucination Detection:** A verdict is flagged as 
potentially hallucinated when LLM confidence exceeds 
τ_conf = 0.75 while maximum retrieved evidence similarity 
falls below τ_sim = 0.45. This condition identifies cases 
where the LLM expresses high certainty despite weak 
evidentiary support — indicating reliance on parametric 
memory rather than retrieved evidence. The Hallucination 
Rate (HR) is formally defined as:

HR = |{claims: sim < τ_sim AND conf > τ_conf}| / |total claims|

This metric constitutes Novel Contribution C3 of 
EvidenceChain, providing a reproducible m

---

### 3.6 Stacking Ensemble Meta-Classifier

EvidenceChain combines the outputs of the RoBERTa classifier 
and RAG reasoning pipeline through a stacking ensemble 
meta-classifier. Rather than averaging scores with fixed 
weights — which assumes equal reliability across all claim 
types — the meta-classifier learns optimal feature 
combinations from validation data.

The meta-classifier receives six input features: (1) RoBERTa 
REAL class probability, (2) RAG verdict confidence, (3) 
maximum evidence similarity score, (4) average source 
credibility weight, (5) hallucination flag, and (6) 
sub-claim count. Features are standardized using 
StandardScaler prior to classification to prevent 
magnitude-based dominance.

Logistic Regression was selected as the meta-classifier 
for two reasons. First, its linear decision boundary is 
appropriate for a small six-dimensional feature space. 
Second, its coefficients are directly interpretable — 
each coefficient quantifies the contribution of its 
corresponding feature to the final verdict, providing 
a natural complement to the SHAP analysis in Section 3.7.

The ensemble is trained on the validation set rather than 
the training set to prevent data leakage — both RoBERTa 
and the RAG pipeline were optimized on training data, 
making validation set features a clean, uncontaminated 
signal for meta-learning.
```

---
## 4. Experimental Setup

### 4.1 Datasets

We evaluate EvidenceChain on two complementary benchmark datasets
selected to test the system across varying claim lengths, domains,
and label granularities.

**Dataset Analysis:** Exploratory analysis of the LIAR training 
split reveals several properties relevant to system design. 
The dataset contains 10,269 training instances with a near-balanced 
binary label distribution (56.2% REAL, 43.8% FAKE), making 
accuracy a viable but insufficient metric — we therefore adopt 
F1-score as our primary evaluation measure. 

Statement lengths range from 2 to 66 words with a mean of 17.9 
words, confirming that LIAR consists of short, dense claims 
well-suited to our decomposition module. Notably, 28% of 
instances contain missing job_title metadata and 21% missing 
state_info, which are imputed with a placeholder value during 
preprocessing. The dataset is dominated by political figures, 
with Barack Obama (493 claims), Donald Trump (274), and 
Hillary Clinton (239) comprising the most frequent speakers, 
reflecting its PolitiFact origin.

**LIAR** [Wang, 2017]: A dataset of 12,800 short political claims
sourced from PolitiFact.com, annotated by human fact-checkers
across six veracity levels: true, mostly-true, half-true,
barely-true, false, and pants-on-fire. Following prior work,
we map these to binary labels: true, mostly-true, and half-true
are mapped to REAL (1); barely-true, false, and pants-on-fire
are mapped to FAKE (0). Crucially, LIAR provides rich speaker
metadata including historical veracity counts, party affiliation,
and statement context, which EvidenceChain incorporates as
additional features in the stacking ensemble.

**ISOT Fake News Dataset** [Ahmed et al., 2020]: A dataset of
44,000 full news articles collected from Reuters (real news)
and websites flagged by PolitiFact as unreliable (fake news),
with binary labels. Unlike LIAR, ISOT contains full article
text, testing the system's ability to handle longer inputs
and stylistic manipulation patterns.
Exploratory analysis of ISOT reveals properties that 
contrast sharply with LIAR. The dataset contains 44,898 
full news articles with a near-balanced binary label 
distribution (52.5% FAKE, 47.5% REAL). Article length 
ranges from 0 to 8,135 words with a mean of 405.7 words 
— approximately 23 times longer than LIAR claims. This 
length difference has direct implications for the 
RoBERTa classifier, which must truncate articles to its 
512-token limit, retaining only the opening portion of 
each article. Notably, ISOT fake news exhibits strong 
stylistic signals — sensational headlines, emotional 
language, and partisan framing — making it a more 
tractable classification task than LIAR's subtle 
political half-truths.

The complementary nature of these datasets is deliberate.
LIAR tests claim-level reasoning on subtle, politically
nuanced statements — the primary target of our decomposition
module. ISOT tests article-level linguistic pattern detection —
the primary target of our RoBERTa classifier. Cross-dataset
evaluation (train on LIAR, test on ISOT and vice versa)
assesses generalization across domains and text lengths.

### 4.2 Data Splits

All datasets are divided using stratified splits to preserve
class distribution: 80% training, 10% validation, 10% test.
Cross-dataset evaluation — training on LIAR and testing on
ISOT, and vice versa — is conducted to assess generalization
across domains.

### 4.3 Baselines

We compare EvidenceChain against the following systems:
(1) TF-IDF + Logistic Regression, a classical baseline;
(2) BERT-only fine-tuned classifier;
(3) RAG-only pipeline without the BERT layer;
(4) SAFE [Zhou et al., 2020], a social-context-aware model;
(5) DFFN [state-of-the-art reference to be added].

### 4.4 Evaluation Metrics

We evaluate using Accuracy, Precision, Recall, F1-score,
and ROC-AUC. Additionally, we introduce three research-specific
metrics: Hallucination Rate (HR), Evidence Recall, and
Faithfulness Score, evaluated using the RAGAS framework.

---

## 5. Results

### 5.1 RoBERTa Baseline Results

Table 1 presents the performance of our fine-tuned RoBERTa-base
classifier on the LIAR test set. The model was trained for 3
epochs with a learning rate of 2e-5 and batch size of 16,
with the best checkpoint selected based on validation F1.

| Model          | Accuracy | F1 (weighted) |
|----------------|----------|---------------|
| RoBERTa-base   | 0.6461   | 0.6443        |

The model achieves 64.6% accuracy and 0.644 weighted F1,
consistent with published BERT-based baselines on the LIAR
dataset which range from 62-68% [Wang, 2017]. Notably, the
model performs better on REAL claims (F1=0.70) than FAKE
claims (F1=0.58), reflecting the inherent difficulty of
detecting subtle misinformation in short political statements.

Validation F1 peaked at Epoch 2 (0.6588) before declining
slightly at Epoch 3 (0.6425), indicating mild overfitting
onset. The best checkpoint from Epoch 2 was used for all
subsequent evaluations.

These results establish our BERT-only baseline. The
subsequent RAG pipeline and stacking ensemble are expected
to improve upon this baseline — improvement magnitude
constitutes our primary research contribution.
```

---

### 5.2 Full Pipeline Results

Table 2 presents qualitative results of the complete 
EvidenceChain pipeline on representative LIAR claims.

| Claim                        | RAG          | BERT | Final | Correct |
|------------------------------|--------------|------|-------|---------|
| COVID vaccines + infertility  | FALSE        | 0.50 | FAKE  | ✅      |
| Unemployment 50-year low      | UNVERIFIABLE | 0.82 | REAL  | ✅      |
| CDC self-destruct             | FALSE        | 0.27 | FAKE  | ✅      |

These results demonstrate the complementary nature of the 
RAG and BERT components. For the unemployment claim, RAG 
returned UNVERIFIABLE due to absence of current data in 
the knowledge base, while RoBERTa correctly classified 
the claim based on linguistic patterns. The ensemble 
correctly deferred to BERT in this case. Conversely, 
for the COVID vaccine claim, BERT was uncertain (p=0.50) 
while RAG produced a confident FALSE verdict supported 
by WHO evidence. The ensemble correctly deferred to RAG.

This complementary behavior validates our stacking 
ensemble design — neither component alone achieves 
correct results across all claim types, but their 
combination produces accurate verdicts in all three cases.
```

---

---

## 6. Ablation Study
*(to be completed on Day 16)*

---

## 7. Human Evaluation
*(to be completed on Day 19)*

---

## 8. Conclusion
*(to be completed on Day 21)*

---

## References
*(added progressively as we cite)*

[Wang, 2017] Wang, W. Y. (2017). "Liar, Liar Pants on Fire":
A New Benchmark Dataset for Fake News Detection. ACL 2017.

[Liu et al., 2019] Liu, Y., et al. RoBERTa: A Robustly Optimized
BERT Pretraining Approach. arXiv:1907.11692.

[Ahmed et al., 2020] Ahmed, H., et al. Detecting opinion spams
and fake news using text classification. Security and Privacy.

[Zhou et al., 2020] Zhou, X., et al. SAFE: Similarity-Aware
Multi-modal Fake News Detection. PAKDD 2020.
```
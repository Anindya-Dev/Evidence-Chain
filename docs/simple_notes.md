# EvidenceChain - Simple Notes

## What is this project?

This project builds a fake news detection system that does not only say
"fake" or "real". It tries to explain why.

The system does this by combining:

1. a text model (RoBERTa)
2. a retrieval system (FAISS)
3. an LLM for reasoning over evidence
4. a final ensemble that combines the signals

## What problem are we trying to solve?

Normal fake news classifiers have three common problems:

1. They treat a big sentence as one claim even when it contains many
   smaller claims.
2. They do not use real evidence properly.
3. They can sound confident even when the answer is not grounded in
   evidence.

EvidenceChain was designed to reduce these problems.

## Main idea of EvidenceChain

When a claim comes in, the system works like this:

1. Clean the text.
2. Break the claim into smaller sub-claims.
3. Search a knowledge base for relevant evidence.
4. Ask an LLM to judge each sub-claim using only that evidence.
5. Check whether the LLM may be hallucinating.
6. Run RoBERTa on the original claim.
7. Combine both branches into one final verdict.

## What are we using and why?

### LIAR dataset

We use LIAR because it contains short political claims and useful
metadata such as speaker and subject. That makes it a good dataset for
claim-level verification.

### ISOT dataset

We also use ISOT because it contains long news articles. It helps test
whether the project can handle longer inputs and article-level fake news.

Important note:

- ISOT preprocessing is already done.
- ISOT model training is still running.
- So the current completed results are mainly for LIAR.

### RoBERTa

We use RoBERTa as the main text-only baseline because it is a strong
transformer model for classification. It gives us a solid benchmark to
beat.

### MiniLM embeddings

We use `all-MiniLM-L6-v2` to turn claims and evidence into vectors. It
is fast and works well for semantic similarity.

### FAISS

We use FAISS because it is a simple and fast way to search through
vector embeddings.

### Groq + Llama 3.3 70B

We use an LLM to do two jobs:

1. break claims into smaller verifiable parts
2. reason over evidence and give a verdict

### Logistic Regression ensemble

We use a small and interpretable final model to combine:

- RoBERTa probability
- RAG confidence
- retrieval similarity
- source credibility
- hallucination flag
- number of sub-claims

## What is special about this project?

### C1 - Claim decomposition

Instead of checking one long sentence as one unit, we split it into
smaller facts.

Example:

"COVID vaccines cause infertility and are banned in Europe"

This becomes:

1. COVID vaccines cause infertility
2. COVID vaccines are banned in Europe

This is useful because one part can be true and another part can be
false.

### C2 - Better evidence ranking

We do not rank evidence only by similarity.

We also look at:

- source credibility
- recency

So a recent Reuters or WHO document should matter more than a weak
source.

### C3 - Hallucination check

If the LLM gives a high-confidence answer but the retrieved evidence is
weak, we flag that output as suspicious.

This is a practical way to track grounding quality.

## What has already been completed?

### Completed now

- LIAR loading and preprocessing
- ISOT loading and preprocessing
- LIAR RoBERTa baseline training
- FAISS knowledge base building
- claim decomposition
- evidence retrieval
- LLM reasoning
- hallucination flagging
- full LIAR pipeline
- evaluation and ablation scripts

### Still pending

- finish ISOT RoBERTa training
- retrain the ensemble on real validation features
- rerun full evaluation after the latest code fixes
- rerun ablation after ensemble retraining
- add SHAP explainability
- run human evaluation

## What do the current saved results show?

### RoBERTa baseline on LIAR

- Accuracy: 0.6461
- Weighted F1: 0.6443

This means the text-only baseline is already fairly strong.

### Current full pipeline sample on LIAR

- Accuracy: 0.5000
- Weighted F1: 0.4833
- ROC-AUC: 0.6208
- Hallucination rate: 0.46

This means the full architecture is working, but it still needs
improvement before it can beat the RoBERTa baseline consistently.

## Why is that okay in a research project?

Because research is not only about getting a better number quickly.

It is also about:

- building a clear pipeline
- understanding where the system fails
- showing which parts help and which parts still need work

Right now the project has reached the point where the architecture is
implemented and the main weakness is clear: the retrieval and fusion
parts need to become stronger.

## What should happen next?

1. Finish ISOT training.
2. Save the ISOT results.
3. Train the ensemble on real validation-set features.
4. Expand the evidence base.
5. Rerun evaluation and ablation.
6. Add SHAP and human evaluation.

## One-line explanation for anyone

EvidenceChain is a fake news detector that breaks a claim into smaller
facts, finds supporting evidence, checks whether the AI is grounded, and
then combines that evidence-based reasoning with a strong RoBERTa
baseline.

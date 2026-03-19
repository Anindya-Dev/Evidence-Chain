# EvidenceChain — Simple Notes

## Title

This project builds a smart system to detect fake news by:

* Breaking claims into smaller parts
* Finding real evidence from trusted sources
* Checking how recent and reliable the evidence is
* Giving a final answer with explanation

---

## 1. Introduction

Fake news is spreading very fast on the internet and social media.
This is dangerous because it can:

* Mislead people
* Affect elections
* Harm people’s health

Even though many fake news detection systems exist, they still have 3 major problems:

### Problem 1 — Multiple Claims Problem

Some sentences contain multiple claims, but systems treat them as one.

Example:
“Vaccines cause infertility and are banned in Europe”

This has 2 claims:

1. Vaccines cause infertility
2. Vaccines are banned in Europe

Current systems cannot separate them, so they may give wrong results.

---

### Problem 2 — All Sources Treated Equal

Systems treat all information sources the same.

But in reality:

* Reuters = highly reliable
* Random blog = not reliable

Also, systems don’t consider time.

Old information may be wrong today.

---

### Problem 3 — AI Hallucination Problem

AI models sometimes give answers without real evidence.

They may:

* Guess answers
* Use old memory
* Sound confident but be wrong

This is called hallucination, and current systems do not measure it.

---

## Our Solution — EvidenceChain

We solve these problems using 3 ideas:

### C1 — Break Claims

We break a big claim into smaller parts and check each separately.

---

### C2 — Smart Evidence Ranking

We give importance to:

* Recent information
* Trusted sources

---

### C3 — Hallucination Check

We check if the AI answer is actually supported by real evidence.

We also measure this using a metric called:
Hallucination Rate (HR)

---

## Models Used

We combine:

* RoBERTa (language understanding model)
* Ensemble model (combining multiple outputs)

Datasets used:

* LIAR dataset
* ISOT dataset

---

## 3. Methodology (Simple)

The system works in steps:

1. Clean the input text
2. Break the claim into smaller parts
3. Find related evidence
4. Let AI analyze evidence
5. Check for hallucination
6. Use BERT/RoBERTa for text classification
7. Combine results
8. Generate explanation

---

## 3.2 Configuration (Simple)

* We fix random seed = 42 (for same results every time)
* We use RoBERTa instead of BERT because it performs better
* We use MiniLM model for sentence similarity
* We retrieve top 5 most relevant pieces of evidence

---

### Hallucination Detection Rule

We use two conditions:

* Evidence similarity must be ≥ 0.45
* AI confidence must be ≤ 0.75

If not, we mark it as unreliable

---

## 4. Experimental Setup

### Datasets

## 4.1 Datasets

We test our system using two different datasets to check how well it works on:

* Short claims
* Long news articles
* Different types of content

---

### LIAR Dataset

* Contains 12,800 short political statements
* Collected from PolitiFact
* Each statement is checked by humans

It has 6 labels:

* true
* mostly true
* half true
* barely true
* false
* pants on fire

We convert these into 2 classes:

REAL (1):

* true
* mostly true
* half true

FAKE (0):

* barely true
* false
* pants on fire

---

### Extra Feature (Important)

LIAR also gives extra information like:

* Who said the statement
* Their political party
* Their past truth history

We use this as additional input in our final model (ensemble).

---

### ISOT Dataset (Simple)

* Contains 44,000 full news articles
* Real news from Reuters
* Fake news from unreliable websites

Unlike LIAR:

* This dataset has full articles (long text)

This helps test:

* Writing style
* Fake news patterns
* Manipulation in long content

---

### Why We Use Both Datasets

We use both because they test different things:

LIAR:

* Short statements
* Needs logical reasoning
* Helps test claim decomposition

ISOT:

* Long articles
* Needs language understanding
* Helps test RoBERTa model

---

### Cross Testing (Very Important)

We also test like this:

* Train on LIAR → Test on ISOT
* Train on ISOT → Test on LIAR

This checks:
👉 Can our model work on new types of data?

---

## Final Understanding

LIAR → tests reasoning on small claims
ISOT → tests understanding of full articles

Using both makes our system more strong and reliable


### Data Split

* 80% training
* 10% validation
* 10% testing

We also test:

* Train on LIAR → Test on ISOT
* Train on ISOT → Test on LIAR

---

### Baselines (Comparison Models)

We compare our model with:

1. TF-IDF + Logistic Regression
2. BERT-only model
3. RAG-only system
4. SAFE model
5. Other state-of-the-art models

---

### Evaluation Metrics

We measure performance using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Extra important metrics:

* Hallucination Rate → how often AI gives unsupported answers
* Evidence Recall → did we retrieve correct evidence
* Faithfulness Score → does answer match evidence

---

## Data Preprocessing 
Before giving the data to our models, we clean and prepare it properly.

---

### Text Cleaning

We apply the following steps:

* Convert all text to lowercase
* Remove URLs (links)
* Remove special characters (except apostrophes like "n't")
* Keep numbers (because they are important in claims)

---

### Why Keep Apostrophes?

Because they help in meaning:

Example:

* "is not" vs "isn't"

Both show negation, which is very important for detecting fake news.

---

### Special Case Handling

Some claims in the LIAR dataset start with the word "Says".

Example:
"Says vaccines are harmful"

We remove "Says" so that we only keep the actual claim.

---

### Handling Missing Data

Some fields are missing, like:

* job title (28%)
* state info (21%)
* context (1%)

Instead of leaving them empty, we replace them with:
"unknown"

This helps the model understand that the data is missing.

---

### Stopwords (Important Decision)

We do NOT remove stopwords like:

* not
* is
* the

Because:

* Words like "not" are very important for meaning
* RoBERTa understands full sentences better when all words are present

---

### Final Input Format

We combine all information into one structured format:

[CLAIM] claim_text
[SPEAKER] speaker_name
[SUBJECT] subject

---

### Example

[CLAIM] vaccines are not safe
[SPEAKER] John Doe
[SUBJECT] health

---

### Why This Helps

This format allows the model to:

* Understand the claim
* Consider who said it
* Understand the topic

So the model makes better predictions using both text and context.

---

## Final Understanding

Data preprocessing helps:

* Clean the text
* Keep important meaning
* Handle missing data properly
* Add extra useful information

This improves the overall performance of the system.

## Dataset Analysis

We analyzed the LIAR dataset to understand its properties before using it.

---

### Size and Label Distribution

* Total training samples: 10,269
* REAL: 56.2%
* FAKE: 43.8%

This means the dataset is almost balanced.

---

### Why Not Only Accuracy?

Since the data is slightly imbalanced, accuracy alone is not enough.

So we use:
👉 F1-score (better metric for balanced evaluation)

---

### Statement Length

* Minimum length: 2 words
* Maximum length: 66 words
* Average length: ~18 words

This shows:
👉 The dataset contains short and compact claims

This is good for our system because:
👉 It works well with claim decomposition

---

### Missing Data

Some metadata is missing:

* job_title missing in 28% cases
* state_info missing in 21% cases

We replace missing values with:
👉 "unknown"

---

### Speakers in Dataset

Most claims are from political figures:

* Barack Obama → 493 claims
* Donald Trump → 274 claims
* Hillary Clinton → 239 claims

This shows:
👉 The dataset mainly focuses on political statements

---

## Final Understanding

* Data is nearly balanced
* Claims are short and suitable for analysis
* Some metadata is missing but handled properly
* Dataset is focused on political figures

This helps us design our system more effectively.

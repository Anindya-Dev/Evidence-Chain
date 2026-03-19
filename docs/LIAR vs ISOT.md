## What are LIAR AND ISOT Dataset

LIAR Dataset
├── Created by : William Yang Wang, UC Santa Barbara (2017)
├── Source     : PolitiFact.com — real human fact-checked claims
├── Size       : 12,800 short political statements
├── Label type : 6-class (true, mostly-true, half-true,
│                barely-true, false, pants-fire)
└── Example    : "The unemployment rate is the lowest
                  it has been in 50 years" — Barack Obama

ISOT Dataset
├── Created by : University of Victoria (2020)
├── Source     : Reuters (real) + PolitiFact flagged sites (fake)
├── Size       : 44,000 full news articles
├── Label type : Binary (real / fake)
└── Example    : Full 500-word article about election fraud

## Key Differences 

┌─────────────────────┬──────────────────────┬───────────────────────┐
│ Property            │ LIAR                 │ ISOT                  │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Text length         │ Short claims         │ Full articles         │
│                     │ 1-2 sentences        │ 300-800 words         │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Label granularity   │ 6 classes → binary   │ Already binary        │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Domain              │ Political statements │ General news          │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Metadata            │ Rich — speaker,      │ Minimal — title,      │
│                     │ party, context,      │ text, subject         │
│                     │ job title, state     │                       │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Challenge           │ Subtle language,     │ Stylistic differences │
│                     │ half-truths          │ easier to detect      │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Best tests          │ Claim decomposition  │ BERT linguistic       │
│                     │ RAG reasoning        │ pattern detection     │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Size                │ Small — 12,800       │ Large — 44,000        │
└─────────────────────┴──────────────────────┴───────────────────────┘

## Why we use both datasets:


LIAR alone  → good for claim-level reasoning
              but small, only political domain

ISOT alone  → good for article-level classification
              but too easy, stylistically obvious

LIAR + ISOT → proves that our system works on:
              ✓ Short claims AND long articles
              ✓ Subtle half-truths AND obvious fake news
              ✓ Political domain AND general news domain

Cross-dataset test:
  Train on LIAR → Test on ISOT = generalization test
  Train on ISOT → Test on LIAR = domain transfer test


## LIAR Columns


Column 1  : ID
            Unique identifier for each claim
            Example: 2635.json

Column 2  : label
            One of 6 values — we map to binary
            This is our TARGET variable (what we predict)

Column 3  : statement
            The actual claim text
            This is our PRIMARY INPUT to the model

Column 4  : subject
            Topic category — healthcare, economy, taxes etc
            Useful as an additional feature

Column 5  : speaker
            Who made the claim — Barack Obama, Donald Trump etc
            Could be used for speaker credibility scoring

Column 6  : speaker_job
            Job title at time of claim — President, Senator etc

Column 7  : state_info
            US state associated with the speaker

Column 8  : party_affiliation
            democrat, republican, none etc

Column 9  : barely_true_count
            How many times this speaker was rated barely-true
            historically — THIS IS GOLD for credibility scoring

Column 10 : false_count
            Historical false count for this speaker

Column 11 : half_true_count
            Historical half-true count for this speaker

Column 12 : mostly_true_count
            Historical mostly-true count for this speaker

Column 13 : pants_on_fire_count
            Historical pants-on-fire count for this speaker

Column 14 : context
            Where the claim was made — a speech, interview, tweet

  
## Important columns [9-13]

Speaker history = credibility signal

If a speaker has:
  false_count      = 45
  pants_fire_count = 12
  true_count       = 3

→ Their new claim is statistically more likely to be fake

This is a feature NO basic BERT classifier uses.

## ISOT columns

Column 1 : title      — headline of the article
Column 2 : text       — full article body
Column 3 : subject    — topic category
Column 4 : date       — publication date
Column 5 : label      — real / fake (binary, already)


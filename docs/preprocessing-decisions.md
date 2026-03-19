Every Preprocessing Decision

1. Lowercase everything
"The Vaccine Is DANGEROUS" → "the vaccine is dangerous"

Why: Models treat "Vaccine" and "vaccine" as different tokens.
Lowercasing removes this artificial difference.
Exception: RoBERTa's tokenizer handles this internally anyway,
but we lowercase for consistency across all modules.

2. Remove URLs
"Check this www.fakeclaims.com for proof" → "Check this for proof"

Why: URLs carry no semantic meaning for classification.
They also never appear in test claims the same way twice.

3. Remove special characters — but keep apostrophes
"@Breaking!! Trump says #economy is great!!!" 
→ "Breaking Trump says economy is great"

Why: Punctuation like @, #, ! adds noise.
Apostrophes kept because "isn't" ≠ "is not" in meaning.

4. Keep numbers
"unemployment rate dropped 50 percent in 10 years"
→ keep as is

Why: Numbers in political claims carry factual meaning.
"50 percent" vs "5 percent" is the difference between
true and false. Removing numbers loses critical signal.

5. Strip "Says" prefix
"Says the Annies List political group supports abortions"
→ "The Annies List political group supports abortions"

Why: LIAR claims frequently start with "Says" — it is
an artifact of PolitiFact's writing style, not part of
the actual claim. It adds no classification signal.

6. Fill missing values with "unknown"
job_title  : NaN → "unknown"
state_info : NaN → "unknown"
context    : NaN → "unknown"

Why: Empty string "" is invisible to the model — it
contributes nothing. The word "unknown" is an explicit
signal that this information was unavailable, which
itself can be a weak feature (unverified speakers tend
to have missing metadata).

7. Normalize whitespace
"The   economy    is    good" → "The economy is good"

Why: Multiple spaces are tokenized differently by some
models. Normalizing ensures clean consistent input.

8. Do NOT remove stopwords
"The vaccine is not dangerous" — keep "not"

Why: — "not" completely flips meaning.
RoBERTa reads full context so stopwords are meaningful.
Removing them would hurt BERT performance.

9. Build combined text field
statement + speaker + subject → one input string

Why: Speaker identity and subject carry signal.
"Barack Obama says X" is processed differently than
"unknown blog says X". Combining gives BERT full context.

Format:
"[CLAIM] vaccines cause autism [SPEAKER] barack-obama [SUBJECT] health"

10. Add binary label column
"false"     → 0
"pants-fire"→ 0
"true"      → 1
"half-true" → 1

Why: Model needs numeric targets not string labels.


### Preprocessing Steps:

Step 1 : Lowercase
Step 2 : Fill missing values  ← where exactly?
Step 3 : Remove URLs
Step 4 : Remove special characters (keep apostrophes)
Step 5 : Strip "Says" prefix
Step 6 : Normalize whitespace
Step 7 : Build combined text field
Step 8 : Add binary label
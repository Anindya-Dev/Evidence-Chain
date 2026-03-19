# Cleans and prepares LIAR dataset for model input.
# Every decision documented with reasoning.

import re
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def preprocess_text(text):
    """
    Cleans a single text string through 5 sequential steps.
    Order matters — lowercasing before regex ensures
    patterns like 'Says' and 'says' are both caught.
    """

    # Safety net — should not reach here after fillna
    # but defensive programming prevents silent failures
    if pd.isna(text):
        return "unknown"

    # Step 1: Lowercase
    # Removes artificial token differences like Vaccine vs vaccine
    text = text.lower()

    # Step 2: Remove URLs
    # http/https/www patterns add noise, never repeat identically
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove special characters, keep apostrophes
    # Apostrophes preserved → "isn't" ≠ "is" for BERT context
    # Numbers preserved → "50 percent" carries factual meaning
    text = re.sub(r"[^a-z0-9\s']", '', text)

    # Step 4: Strip "says" prefix
    # PolitiFact artifact — not part of the actual claim
    # ^ anchors to start of string only
    text = re.sub(r'^says\s+', '', text)

    # Step 5: Normalize whitespace
    # Collapses multiple spaces into one clean space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataframe(df):
    """
    Applies full preprocessing pipeline to a DataFrame.
    Returns enriched DataFrame ready for model input.
    """

    # Work on a copy — never mutate original data
    # This lets us compare raw vs processed if needed
    df = df.copy()

    # Step 1: Fill missing values in text columns only
    # Why "unknown" not ""?
    # "unknown" is an explicit signal — empty string is invisible
    # Numeric columns (barely_true_counts etc.) left untouched
    text_cols = ["job_title", "state_info", "context",
                 "speaker", "subject", "party_affiliation"]
    for col in text_cols:
        df[col] = df[col].fillna("unknown").replace("", "unknown")

    # Step 2: Apply text cleaning to statement column
    df["clean_statement"] = df["statement"].apply(preprocess_text)

    # Step 3: Build structured combined text field
    # Why structured tags [CLAIM] [SPEAKER] [SUBJECT]?
    # RoBERTa can attend to these boundaries during fine-tuning
    # Speaker context helps — Obama's claims ≠ unknown blog's claims
    df["combined_text"] = (
        "[CLAIM] "   + df["clean_statement"] +
        " [SPEAKER] " + df["speaker"] +
        " [SUBJECT] " + df["subject"]
    )

    # Step 4: Map 6-class labels to binary using config
    # Why config.LABEL_MAP and not hardcoded?
    # Single source of truth — change mapping in one place only
    df["binary_label"] = df["label"].map(config.LABEL_MAP)

    return df


if __name__ == "__main__":
    from data_loader import load_liar

    train_df, val_df, test_df = load_liar()

    print("Preprocessing train split...")
    train_clean = preprocess_dataframe(train_df)

    # Save processed splits to data/processed/
    # Why save? Preprocessing takes time — load clean data directly next time
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    val_clean  = preprocess_dataframe(val_df)
    test_clean = preprocess_dataframe(test_df)

    train_clean.to_csv(os.path.join(config.DATA_PROCESSED, "train.csv"), index=False)
    val_clean.to_csv(os.path.join(config.DATA_PROCESSED,   "val.csv"),   index=False)
    test_clean.to_csv(os.path.join(config.DATA_PROCESSED,  "test.csv"),  index=False)

    print("\nProcessed files saved to data/processed/")
    print(f"  Train : {len(train_clean)} rows")
    print(f"  Val   : {len(val_clean)} rows")
    print(f"  Test  : {len(test_clean)} rows")



    # Verify output
    print("\nSample output:")
    for i in range(3):
        row = train_clean.iloc[i]
        print(f"\nRow {i+1}:")
        print(f"  Original  : {train_df.iloc[i]['statement']}")
        print(f"  Cleaned   : {row['clean_statement']}")
        print(f"  Combined  : {row['combined_text']}")
        print(f"  Label     : {row['label']} → {row['binary_label']}")
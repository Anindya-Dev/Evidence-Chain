# modules/preprocessor.py
# Cleans and prepares LIAR and ISOT datasets for model input.
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

    if pd.isna(text):
        return "unknown"

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove special characters, keep apostrophes
    text = re.sub(r"[^a-z0-9\s']", '', text)

    # Step 4: Strip "says" prefix — PolitiFact artifact
    text = re.sub(r'^says\s+', '', text)

    # Step 5: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataframe(df):
    """
    Preprocesses LIAR DataFrame.
    Returns enriched DataFrame ready for model input.
    """

    df = df.copy()

    # Fill missing text columns with "unknown"
    text_cols = ["job_title", "state_info", "context",
                 "speaker", "subject", "party_affiliation"]
    for col in text_cols:
        df[col] = df[col].fillna("unknown").replace("", "unknown")

    # Clean statement
    df["clean_statement"] = df["statement"].apply(preprocess_text)

    # Build structured combined text
    df["combined_text"] = (
        "[CLAIM] "    + df["clean_statement"] +
        " [SPEAKER] " + df["speaker"] +
        " [SUBJECT] " + df["subject"]
    )

    # Map 6-class to binary
    df["binary_label"] = df["label"].map(config.LABEL_MAP)

    return df


def preprocess_isot_dataframe(df):
    """
    Preprocesses ISOT DataFrame.

    Key difference from LIAR:
    ISOT has full articles — we use title + first 400 words.
    Why 400 words? RoBERTa max is 512 tokens.
    Taking title + opening captures the most important content.
    News articles bury key claims in the first paragraph.
    """

    df = df.copy()

    # Clean title
    df["title"] = df["title"].fillna("").apply(preprocess_text)

    # Truncate text to first 400 words then clean
    # Why 400? Leaves room for title within 512 token limit
    df["text_truncated"] = df["text"].fillna("").apply(
        lambda x: " ".join(str(x).split()[:400])
    )
    df["text_truncated"] = df["text_truncated"].apply(preprocess_text)

    # Build combined text — title carries strong signal in ISOT
    # Sensational titles vs neutral Reuters titles
    df["combined_text"] = (
        "[TITLE] " + df["title"] +
        " [BODY] "  + df["text_truncated"]
    )

    return df


if __name__ == "__main__":
    from data_loader import load_liar, load_isot

    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    # ── LIAR ──────────────────────────────────────────────
    print("Loading and preprocessing LIAR...")
    train_df, val_df, test_df = load_liar()

    train_clean = preprocess_dataframe(train_df)
    val_clean   = preprocess_dataframe(val_df)
    test_clean  = preprocess_dataframe(test_df)

    train_clean.to_csv(os.path.join(config.DATA_PROCESSED, "train.csv"), index=False)
    val_clean.to_csv(os.path.join(config.DATA_PROCESSED,   "val.csv"),   index=False)
    test_clean.to_csv(os.path.join(config.DATA_PROCESSED,  "test.csv"),  index=False)

    print(f"  Train : {len(train_clean)} rows saved")
    print(f"  Val   : {len(val_clean)} rows saved")
    print(f"  Test  : {len(test_clean)} rows saved")

    # Verify LIAR
    print("\nSample LIAR output:")
    for i in range(2):
        row = train_clean.iloc[i]
        print(f"\n  Row {i+1}:")
        print(f"  Original : {train_df.iloc[i]['statement']}")
        print(f"  Cleaned  : {row['clean_statement']}")
        print(f"  Label    : {row['label']} → {row['binary_label']}")

    # ── ISOT ──────────────────────────────────────────────
    print("\n\nLoading and preprocessing ISOT...")
    isot_train, isot_val, isot_test = load_isot()

    isot_train_clean = preprocess_isot_dataframe(isot_train)
    isot_val_clean   = preprocess_isot_dataframe(isot_val)
    isot_test_clean  = preprocess_isot_dataframe(isot_test)

    isot_train_clean.to_csv(os.path.join(config.DATA_PROCESSED, "isot_train.csv"), index=False)
    isot_val_clean.to_csv(os.path.join(config.DATA_PROCESSED,   "isot_val.csv"),   index=False)
    isot_test_clean.to_csv(os.path.join(config.DATA_PROCESSED,  "isot_test.csv"),  index=False)

    print(f"  Train : {len(isot_train_clean)} rows saved")
    print(f"  Val   : {len(isot_val_clean)} rows saved")
    print(f"  Test  : {len(isot_test_clean)} rows saved")

    # Verify ISOT
    print("\nSample ISOT output:")
    for i in range(2):
        row = isot_train_clean.iloc[i]
        print(f"\n  Row {i+1}:")
        print(f"  Label    : {row['label']} → {row['binary_label']}")
        print(f"  Combined : {row['combined_text'][:150]}...")

    print("\nAll preprocessing complete.")
    print(f"Files in data/processed/: {os.listdir(config.DATA_PROCESSED)}")
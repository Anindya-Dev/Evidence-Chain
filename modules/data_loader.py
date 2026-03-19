# modules/data_loader.py
# Responsible for loading LIAR dataset from local TSV files
# and returning clean pandas DataFrames.
#
# Why local TSV and not HuggingFace loader?
# HuggingFace deprecated script-based dataset loading.
# Loading from original source TSV files is more stable
# and gives us full control over the data pipeline.

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add root folder to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

load_dotenv()


def load_liar():
    """
    Loads the LIAR dataset from local TSV files.

    LIAR original format:
    - Tab separated, no header row
    - 14 columns per row
    - 3 splits: train, valid, test

    Returns:
        train_df, val_df, test_df as pandas DataFrames
    """

    print("Loading LIAR dataset from local files...")

    # Read each split using official column names from dataset script
    # sep="\t"     → tab separated
    # header=None  → no header row in file
    # names=...    → we provide column names manually
    train_df = pd.read_csv(
        os.path.join(config.DATA_RAW, "train.tsv"),
        sep="\t",
        header=None,
        names=config.LIAR_COLUMNS,
        quoting=3       # quoting=3 = QUOTE_NONE, handles special chars
    )

    val_df = pd.read_csv(
        os.path.join(config.DATA_RAW, "valid.tsv"),
        sep="\t",
        header=None,
        names=config.LIAR_COLUMNS,
        quoting=3
    )

    test_df = pd.read_csv(
        os.path.join(config.DATA_RAW, "test.tsv"),
        sep="\t",
        header=None,
        names=config.LIAR_COLUMNS,
        quoting=3
    )

    print(f"  Train size : {len(train_df)}")
    print(f"  Val size   : {len(val_df)}")
    print(f"  Test size  : {len(test_df)}")
    print(f"  Columns    : {list(train_df.columns)}")

    return train_df, val_df, test_df


def explore_liar(df, split_name="train"):
    """
    Prints a deep exploration of the LIAR dataset.
    Run this ONCE to understand your data before preprocessing.
    Understanding data = understanding your research problem.

    Args:
        df         : DataFrame to explore
        split_name : name of the split for display purposes
    """

    print(f"\n{'='*55}")
    print(f"  LIAR Dataset — {split_name} split")
    print(f"{'='*55}")

    # Shape — how many rows and columns
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Label distribution — are classes balanced?
    # Imbalanced classes affect model training and metric choice
    print(f"\nLabel distribution (original 6-class):")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:<15} : {count:>4}  ({pct:.1f}%)")

    # After binary mapping — what does our target look like?
    df["binary_label"] = df["label"].map(config.LABEL_MAP)
    print(f"\nLabel distribution (binary):")
    binary_counts = df["binary_label"].value_counts()
    for label, count in binary_counts.items():
        name = "REAL" if label == 1 else "FAKE"
        pct = count / len(df) * 100
        print(f"  {name} ({label})        : {count:>4}  ({pct:.1f}%)")

    # Missing values — affects preprocessing decisions
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col:<25} : {count} missing")
    if missing.sum() == 0:
        print("  No missing values found")

    # Statement length — important for understanding
    # whether BERT's 512 token limit is sufficient
    df["statement_length"] = df["statement"].str.split().str.len()
    print(f"\nStatement length (words):")
    print(f"  Min    : {df['statement_length'].min()}")
    print(f"  Max    : {df['statement_length'].max()}")
    print(f"  Mean   : {df['statement_length'].mean():.1f}")
    print(f"  Median : {df['statement_length'].median():.1f}")

    # Top speakers — who makes the most claims?
    print(f"\nTop 5 speakers:")
    top_speakers = df["speaker"].value_counts().head(5)
    for speaker, count in top_speakers.items():
        print(f"  {speaker:<30} : {count} claims")

    # Sample statements — read a few to feel the data
    print(f"\nSample statements (one per label):")
    for label in df["label"].unique():
        sample = df[df["label"] == label].iloc[0]
        print(f"\n  [{label}]")
        print(f"  Speaker : {sample['speaker']}")
        print(f"  Claim   : {sample['statement']}")


if __name__ == "__main__":
    # Run this file directly to load and explore the dataset
    # Command: python modules/data_loader.py

    train_df, val_df, test_df = load_liar()
    explore_liar(train_df, "train")
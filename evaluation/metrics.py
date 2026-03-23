# evaluation/metrics.py
# Computes all evaluation metrics for EvidenceChain.
# Every metric here was promised in Section 4.4 of the paper.
#
# Why a separate evaluation module?
# Clean separation — pipeline does prediction, this does measurement.
# Also allows evaluating individual components in isolation.

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _resolve_bert_results_path():
    """
    Prefer the dataset-specific BERT results file.
    Fall back to the legacy LIAR file if that is the only artifact present.
    """

    preferred = os.path.join(
        config.TABLES_DIR, f"bert_results_{config.DATASET}.csv"
    )
    if os.path.exists(preferred):
        return preferred

    legacy = os.path.join(config.TABLES_DIR, "bert_results.csv")
    return legacy


def compute_classification_metrics(labels, predictions, probabilities=None):
    """
    Computes standard classification metrics.
    
    Args:
        labels       : true binary labels (0/1)
        predictions  : predicted binary labels (0/1)
        probabilities: predicted probabilities for ROC-AUC
        
    Returns:
        dict of all metrics
    """

    metrics = {
        "accuracy"  : round(accuracy_score(labels, predictions), 4),
        "f1"        : round(f1_score(labels, predictions, average="weighted"), 4),
        "precision" : round(precision_score(labels, predictions, average="weighted"), 4),
        "recall"    : round(recall_score(labels, predictions, average="weighted"), 4),
    }

    # ROC-AUC requires probabilities
    if probabilities is not None:
        try:
            metrics["roc_auc"] = round(
                roc_auc_score(labels, probabilities), 4
            )
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def compute_hallucination_rate(pipeline_results):
    """
    Computes Hallucination Rate — Novel Metric C3.
    
    HR = claims flagged as hallucinated / total claims
    
    A claim is hallucinated when:
    LLM confidence > 0.75 AND evidence similarity < 0.45
    
    Args:
        pipeline_results : list of pipeline.verify() outputs
        
    Returns:
        float — hallucination rate (0 to 1)
    """

    total      = len(pipeline_results)
    hallucinated = sum(
        1 for r in pipeline_results if r.get("hallucinated", False)
    )

    hr = hallucinated / total if total > 0 else 0.0
    return round(hr, 4)


def compute_evidence_recall(pipeline_results, relevant_threshold=0.45):
    """
    Computes Evidence Recall — did we retrieve relevant evidence?
    
    Evidence Recall = claims with at least one strong retrieved
                      evidence / total claims
                      
    Strong evidence = similarity score > relevant_threshold
    
    Args:
        pipeline_results  : list of pipeline outputs
        relevant_threshold: minimum similarity for relevant evidence
        
    Returns:
        float — evidence recall rate
    """

    total    = len(pipeline_results)
    relevant = 0

    for r in pipeline_results:
        # Count a claim as successfully grounded only when at least
        # one sub-claim had genuinely strong retrieval similarity.
        sub_results = r.get("sub_results", [])
        if any(
            sr.get("max_similarity", 0.0) >= relevant_threshold
            for sr in sub_results
        ):
            relevant += 1

    return round(relevant / total, 4) if total > 0 else 0.0


def plot_confusion_matrix(labels, predictions, save_path=None):
    """
    Plots and saves confusion matrix for paper.
    
    Why confusion matrix in paper?
    Shows exactly where the model fails —
    which class gets confused with which.
    """

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels = config.LABEL_NAMES,
        yticklabels = config.LABEL_NAMES
    )
    plt.title("EvidenceChain — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_score_distribution(pipeline_results, save_path=None):
    """
    Plots distribution of confidence scores by verdict.
    Shows how confident the system is across different verdicts.
    """

    verdicts     = [r["final_verdict"] for r in pipeline_results]
    confidences  = [r["confidence"] for r in pipeline_results]

    df = pd.DataFrame({
        "verdict"   : verdicts,
        "confidence": confidences
    })

    plt.figure(figsize=(8, 5))
    for verdict in df["verdict"].unique():
        subset = df[df["verdict"] == verdict]["confidence"]
        plt.hist(subset, alpha=0.6, label=verdict, bins=15)

    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("EvidenceChain — Confidence Distribution by Verdict")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Score distribution saved to {save_path}")

    plt.show()


def generate_results_table(metrics_dict, save_path=None):
    """
    Generates results comparison table for paper.
    Compares all systems side by side.
    """

    df = pd.DataFrame(metrics_dict).T
    df.index.name = "System"

    print("\n" + "="*65)
    print("  RESULTS TABLE — EvidenceChain vs Baselines")
    print("="*65)
    print(df.to_string())

    if save_path:
        df.to_csv(save_path)
        print(f"\nResults table saved to {save_path}")

    return df


def run_full_evaluation(pipeline, test_df, sample_size=None):
    """
    Runs complete evaluation on test set.
    
    Args:
        pipeline    : EvidenceChain instance
        test_df     : preprocessed test DataFrame
        sample_size : number of samples (None = full test set)
        
    Returns:
        dict of all metrics
    """

    if sample_size:
        test_df = test_df.sample(
            sample_size, random_state=config.RANDOM_SEED
        )

    print(f"Running evaluation on {len(test_df)} claims...")
    print("This may take several minutes.\n")

    all_results  = []
    all_labels   = []
    all_preds    = []
    all_probs    = []

    for i, row in test_df.iterrows():
        try:
            # Run full pipeline
            result = pipeline.verify(row["statement"])

            # Convert verdict to binary
            pred  = 1 if result["final_verdict"] == "REAL" else 0
            label = int(row["binary_label"])
            prob  = result["confidence"] if pred == 1 else 1 - result["confidence"]

            all_results.append(result)
            all_labels.append(label)
            all_preds.append(pred)
            all_probs.append(prob)

            # Progress
            if (len(all_results)) % 10 == 0:
                print(f"  Processed {len(all_results)}/{len(test_df)}")

        except Exception as e:
            print(f"  Warning: Failed on claim {i} — {e}")
            continue

    # Compute all metrics
    metrics = compute_classification_metrics(
        all_labels, all_preds, all_probs
    )
    metrics["hallucination_rate"] = compute_hallucination_rate(all_results)
    metrics["evidence_recall"]    = compute_evidence_recall(all_results)
    metrics["n_evaluated"]        = len(all_results)

    # Print results
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<25} : {v}")

    # Save results
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    results_path = os.path.join(config.TABLES_DIR, "full_evaluation.json")
    with open(results_path, "w") as f:
        json.dump({
            "metrics" : metrics,
            "results" : all_results[:10]  # save first 10 for inspection
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return metrics, all_results, all_labels, all_preds


if __name__ == "__main__":
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from pipeline import EvidenceChain

    # Load pipeline
    pipeline = EvidenceChain()

    # Load test data
    test_df = pd.read_csv(os.path.join(config.DATA_PROCESSED, "test.csv"))
    test_df = test_df.dropna(subset=["binary_label", "statement"])

    # Run evaluation on small sample first
    # Change sample_size=None for full evaluation
    metrics, results, labels, preds = run_full_evaluation(
        pipeline   = pipeline,
        test_df    = test_df,
        sample_size = 50        # use 50 for speed, None for full
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        labels, preds,
        save_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    )

    # Plot confidence distribution
    plot_score_distribution(
        results,
        save_path = os.path.join(config.PLOTS_DIR, "confidence_dist.png")
    )

    # Results comparison table
    # BERT baseline from bert_results.csv
    bert_results = pd.read_csv(_resolve_bert_results_path())

    comparison = {
        "BERT-only" : {
            "accuracy"          : bert_results["accuracy"].values[0],
            "f1"                : bert_results["f1_score"].values[0],
            "hallucination_rate": "N/A",
            "evidence_recall"   : "N/A"
        },
        "EvidenceChain" : {
            "accuracy"          : metrics["accuracy"],
            "f1"                : metrics["f1"],
            "hallucination_rate": metrics["hallucination_rate"],
            "evidence_recall"   : metrics["evidence_recall"]
        }
    }

    generate_results_table(
        comparison,
        save_path = os.path.join(config.TABLES_DIR, "comparison_table.csv")
    )

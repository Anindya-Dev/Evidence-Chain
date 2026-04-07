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
import pickle
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


def _should_show_plots():
    """Avoid blocking local experiment runs on interactive plot windows."""

    return not config.CHEAP_MODE


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


def _extract_bert_baseline_metrics(bert_results):
    """
    Normalizes legacy and dataset-specific BERT result schemas.

    Older LIAR artifacts use ``accuracy`` / ``f1_score`` while the newer
    dataset-specific trainer writes ``test_accuracy`` / ``test_f1_score``.
    """

    accuracy_columns = ("accuracy", "test_accuracy")
    f1_columns = ("f1_score", "test_f1_score", "f1")

    accuracy = next(
        (bert_results[col].values[0] for col in accuracy_columns
         if col in bert_results.columns),
        None
    )
    f1 = next(
        (bert_results[col].values[0] for col in f1_columns
         if col in bert_results.columns),
        None
    )

    if accuracy is None or f1 is None:
        raise KeyError(
            "Could not extract baseline metrics from BERT results. "
            f"Available columns: {list(bert_results.columns)}"
        )

    return {
        "accuracy": accuracy,
        "f1": f1
    }


def _resolve_test_data_path():
    """Returns the processed test file for the active dataset."""

    return config.get_processed_split_path("test")


def _load_knowledge_base_summary():
    """Summarizes the active knowledge base for saved evaluation artifacts."""

    kb_dir = config.get_knowledge_base_dir()
    meta_path = os.path.join(kb_dir, "metadata.pkl")
    summary = {
        "knowledge_base_dir": kb_dir,
        "metadata_path": meta_path,
        "metadata_available": os.path.exists(meta_path),
    }

    if not os.path.exists(meta_path):
        return summary

    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        source_counts = {}
        kb_profiles = {}
        provenance_missing = 0
        external_docs = 0
        train_derived_docs = 0

        for item in metadata:
            source = item.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

            profile = item.get("kb_profile", "unknown")
            kb_profiles[profile] = kb_profiles.get(profile, 0) + 1

            if "origin_split" not in item or "is_external" not in item:
                provenance_missing += 1

            if item.get("is_external") is True:
                external_docs += 1
            if str(item.get("origin_split", "")).lower() == "train":
                train_derived_docs += 1

        summary.update({
            "document_count": len(metadata),
            "source_counts": source_counts,
            "kb_profiles": kb_profiles,
            "external_docs": external_docs,
            "train_derived_docs": train_derived_docs,
            "provenance_missing_docs": provenance_missing,
        })
    except Exception as exc:
        summary["metadata_error"] = str(exc)

    return summary


def _build_run_context(full_test_size, requested_count, sample_size):
    """Builds a compact metadata block for saved evaluation outputs."""

    return {
        "dataset": config.DATASET,
        "benchmark_profile": config.get_benchmark_profile(),
        "cheap_mode": config.CHEAP_MODE,
        "strict_review_mode": config.STRICT_REVIEW_MODE,
        "full_test_size": int(full_test_size),
        "requested_count": int(requested_count),
        "sample_size": None if sample_size is None else int(sample_size),
        "save_all_eval_results": config.SAVE_ALL_EVAL_RESULTS,
    }


def _collect_review_warnings(run_context, kb_summary, coverage=None, failure_count=0):
    """Returns benchmark/evaluation caveats that belong in saved artifacts."""

    notes = config.get_benchmark_notes(
        dataset=run_context["dataset"],
        benchmark_profile=run_context["benchmark_profile"]
    )
    warnings = list(notes["notes"])
    severe_issues = list(notes["severe_issues"])

    if run_context["sample_size"] is not None:
        warnings.append(
            f"Evaluation used a sampled subset of {run_context['requested_count']} "
            "examples rather than the full test split."
        )

    if kb_summary.get("provenance_missing_docs", 0):
        warnings.append(
            "Knowledge-base provenance metadata is incomplete. Rebuild the "
            "index to distinguish external evidence from dataset-derived documents."
        )

    if (
        run_context["benchmark_profile"] == "verification" and
        kb_summary.get("train_derived_docs", 0) > 0
    ):
        severe_issues.append(
            "The active knowledge base contains train-split documents, which "
            "is not suitable as primary evidence for an external-verification claim."
        )

    if coverage is not None and coverage < 1.0:
        warnings.append(
            f"Pipeline coverage is {coverage:.4f}; some examples failed during "
            "inference and were not converted into predictions."
        )

    if failure_count:
        warnings.append(
            f"{failure_count} examples failed during inference and are listed "
            "in the saved evaluation artifact."
        )

    return {
        "notes": warnings,
        "severe_issues": severe_issues,
        "review_ready": (
            not severe_issues and
            run_context["sample_size"] is None and
            not config.CHEAP_MODE and
            failure_count == 0
        )
    }


def _resolve_text_column(df):
    """
    Returns the text column that should be passed into pipeline.verify().

    LIAR evaluations use the original claim text in ``statement``.
    ISOT evaluations use the processed article representation in
    ``combined_text`` because there is no ``statement`` column.
    """

    if "statement" in df.columns:
        return "statement"
    if "combined_text" in df.columns:
        return "combined_text"

    raise KeyError(
        "Could not find an evaluation text column. "
        "Expected 'statement' or 'combined_text'."
    )


def _sample_evaluation_df(df, sample_size):
    """
    Returns a reproducible stratified sample when labels are available.

    This keeps cheap-mode evaluations more representative than a plain
    random head/sample, especially on smaller subsets.
    """

    if sample_size is None or sample_size >= len(df):
        return df

    if "binary_label" not in df.columns:
        return df.sample(sample_size, random_state=config.RANDOM_SEED)

    parts = []
    for label, group in df.groupby("binary_label"):
        n = max(1, round(sample_size * len(group) / len(df)))
        n = min(n, len(group))
        parts.append(group.sample(n, random_state=config.RANDOM_SEED))

    sampled = pd.concat(parts).sample(frac=1, random_state=config.RANDOM_SEED)

    if len(sampled) > sample_size:
        sampled = sampled.sample(sample_size, random_state=config.RANDOM_SEED)
    elif len(sampled) < sample_size:
        remaining = df.drop(sampled.index, errors="ignore")
        needed = min(sample_size - len(sampled), len(remaining))
        if needed > 0:
            sampled = pd.concat([
                sampled,
                remaining.sample(needed, random_state=config.RANDOM_SEED)
            ])
            sampled = sampled.sample(frac=1, random_state=config.RANDOM_SEED)

    return sampled.reset_index(drop=True)


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

    if _should_show_plots():
        plt.show()
    else:
        plt.close()


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

    if _should_show_plots():
        plt.show()
    else:
        plt.close()


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


def run_full_evaluation(pipeline, test_df, sample_size=None, text_column=None):
    """
    Runs complete evaluation on test set.
    
    Args:
        pipeline    : EvidenceChain instance
        test_df     : preprocessed test DataFrame
        sample_size : number of samples (None = full test set)
        text_column : column sent into pipeline.verify()
        
    Returns:
        dict of all metrics
    """

    if text_column is None:
        text_column = _resolve_text_column(test_df)

    if sample_size is None and config.CHEAP_MODE:
        sample_size = config.CHEAP_EVAL_SAMPLE_SIZE
        print(
            f"Cheap mode enabled: evaluating a stratified subset of "
            f"{sample_size} examples"
        )

    full_test_size = len(test_df)

    if sample_size:
        test_df = _sample_evaluation_df(test_df, sample_size)

    print(f"Running evaluation on {len(test_df)} claims...")
    print("This may take several minutes.\n")

    requested_count = len(test_df)
    all_results  = []
    all_labels   = []
    all_preds    = []
    all_probs    = []
    failed_cases = []

    for i, row in test_df.iterrows():
        try:
            # Run full pipeline
            result = pipeline.verify(row)

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
            failed_cases.append({
                "row_index": int(i),
                "label": int(row["binary_label"]) if "binary_label" in row else None,
                "claim_preview": str(row.get(text_column, ""))[:200],
                "error": str(e)
            })
            continue

    if not all_results:
        raise RuntimeError("Evaluation produced zero successful predictions.")

    # Compute all metrics
    metrics = compute_classification_metrics(
        all_labels, all_preds, all_probs
    )
    metrics["hallucination_rate"] = compute_hallucination_rate(all_results)
    metrics["evidence_recall"]    = compute_evidence_recall(all_results)
    metrics["n_evaluated"]        = len(all_results)
    metrics["n_requested"]        = requested_count
    metrics["n_failed"]           = len(failed_cases)
    metrics["coverage"]           = round(
        len(all_results) / requested_count, 4
    ) if requested_count else 0.0

    run_context = _build_run_context(
        full_test_size=full_test_size,
        requested_count=requested_count,
        sample_size=sample_size
    )
    kb_summary = _load_knowledge_base_summary()
    review_warnings = _collect_review_warnings(
        run_context,
        kb_summary,
        coverage=metrics["coverage"],
        failure_count=len(failed_cases)
    )

    # Print results
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<25} : {v}")
    if review_warnings["notes"] or review_warnings["severe_issues"]:
        print("\nReviewer notes:")
        for note in review_warnings["notes"]:
            print(f"  - {note}")
        for issue in review_warnings["severe_issues"]:
            print(f"  - SEVERE: {issue}")

    # Save results
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    results_path = os.path.join(
        config.TABLES_DIR, f"full_evaluation_{config.DATASET}.json"
    )
    saved_results = (
        all_results if config.SAVE_ALL_EVAL_RESULTS else all_results[:10]
    )
    with open(results_path, "w") as f:
        json.dump({
            "metrics" : metrics,
            "run_context": run_context,
            "knowledge_base_summary": kb_summary,
            "review_warnings": review_warnings,
            "failures": failed_cases[:50],
            "results" : saved_results
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
    test_df = pd.read_csv(_resolve_test_data_path())
    text_column = _resolve_text_column(test_df)
    test_df = test_df.dropna(subset=["binary_label", text_column])

    # Run evaluation on small sample first
    # Change sample_size=None for full evaluation
    metrics, results, labels, preds = run_full_evaluation(
        pipeline   = pipeline,
        test_df    = test_df,
        sample_size = (
            config.CHEAP_EVAL_SAMPLE_SIZE
            if config.CHEAP_MODE else None
        ),
        text_column = text_column
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
    bert_baseline = _extract_bert_baseline_metrics(bert_results)

    comparison = {
        "BERT-only" : {
            "accuracy"          : bert_baseline["accuracy"],
            "f1"                : bert_baseline["f1"],
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
        save_path = os.path.join(
            config.TABLES_DIR, f"comparison_table_{config.DATASET}.csv"
        )
    )

"""Reviewer-oriented audit report for dataset, artifact, and KB status."""

import json
import os
from datetime import datetime

import pandas as pd

import config
from src.evaluation.metrics import _load_knowledge_base_summary


def _load_split_summary(split):
    """Loads one processed split and returns shape/label metadata."""

    path = config.get_processed_split_path(split)
    if not os.path.exists(path):
        return {
            "path": path,
            "available": False,
        }

    df = pd.read_csv(path)
    label_counts = (
        df["binary_label"].value_counts().sort_index().to_dict()
        if "binary_label" in df.columns else {}
    )
    return {
        "path": path,
        "available": True,
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "columns": list(df.columns),
    }


def _load_saved_artifact_summary():
    """Collects the main saved result files for the active dataset."""

    dataset = config.DATASET
    artifacts = {
        "evaluation_json": os.path.join(
            config.TABLES_DIR, f"full_evaluation_{dataset}.json"
        ),
        "ablation_csv": os.path.join(
            config.TABLES_DIR, f"ablation_results_{dataset}.csv"
        ),
        "comparison_csv": os.path.join(
            config.TABLES_DIR, f"comparison_table_{dataset}.csv"
        ),
        "bert_csv": os.path.join(
            config.TABLES_DIR, f"bert_results_{dataset}.csv"
        ),
    }

    summary = {}
    for name, path in artifacts.items():
        summary[name] = {
            "path": path,
            "available": os.path.exists(path),
        }

    return summary


def build_audit():
    """Builds a reviewer-oriented audit of the current project state."""

    benchmark_notes = config.get_benchmark_notes()
    kb_summary = _load_knowledge_base_summary()
    split_summaries = {
        split: _load_split_summary(split) for split in ("train", "val", "test")
    }
    artifacts = _load_saved_artifact_summary()

    warnings = list(benchmark_notes["notes"])
    severe_issues = list(benchmark_notes["severe_issues"])

    if config.CHEAP_MODE:
        warnings.append(
            "Cheap mode is currently enabled in the environment. Final reported "
            "numbers should be rerun with CHEAP_MODE=false."
        )

    if kb_summary.get("train_derived_docs", 0) > 0:
        if config.get_benchmark_profile() == "verification":
            severe_issues.append(
                "The active knowledge base contains train-derived documents "
                "while the benchmark is configured as verification."
            )
        else:
            warnings.append(
                "The active knowledge base contains train-derived documents. "
                "Treat this as auxiliary retrieval for classification, not "
                "as independent external evidence."
            )

    if kb_summary.get("provenance_missing_docs", 0):
        warnings.append(
            "Knowledge-base provenance metadata is incomplete. Rebuild the KB "
            "to make evidence provenance auditable."
        )

    audit = {
        "generated_at": datetime.now().isoformat(),
        "dataset": config.DATASET,
        "benchmark_profile": config.get_benchmark_profile(),
        "cheap_mode": config.CHEAP_MODE,
        "strict_review_mode": config.STRICT_REVIEW_MODE,
        "split_summaries": split_summaries,
        "knowledge_base_summary": kb_summary,
        "artifacts": artifacts,
        "warnings": warnings,
        "severe_issues": severe_issues,
        "review_ready": (
            not severe_issues and
            not config.CHEAP_MODE
        ),
    }

    return audit


if __name__ == "__main__":
    audit = build_audit()
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    with open(config.REVIEW_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("=" * 65)
    print("  REVIEWER AUDIT")
    print("=" * 65)
    print(f"Dataset           : {audit['dataset']}")
    print(f"Benchmark profile : {audit['benchmark_profile']}")
    print(f"Cheap mode        : {audit['cheap_mode']}")
    print(f"Review ready      : {audit['review_ready']}")
    print(f"Saved to          : {config.REVIEW_AUDIT_PATH}")

    if audit["warnings"]:
        print("\nWarnings:")
        for warning in audit["warnings"]:
            print(f"  - {warning}")

    if audit["severe_issues"]:
        print("\nSevere issues:")
        for issue in audit["severe_issues"]:
            print(f"  - {issue}")

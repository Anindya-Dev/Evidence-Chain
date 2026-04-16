"""Central configuration for the EvidenceChain codebase."""

import os
from dotenv import load_dotenv

load_dotenv()


def _normalize_dataset_name(dataset=None):
    """Returns the active dataset name in a consistent lowercase form."""

    if dataset is None:
        dataset = globals().get("DATASET", "")
    return str(dataset).strip().lower()


def _env_flag(name, default=False):
    """Parses a boolean env var with a safe default."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ── Paths ──────────────────────────────────────────────────────────────
CONFIG_DIR      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.dirname(CONFIG_DIR)

DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
KNOWLEDGE_BASE  = os.path.join(BASE_DIR, "data", "knowledge_base")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(BASE_DIR, "results", "plots")
TABLES_DIR      = os.path.join(BASE_DIR, "results", "tables")
MODELS_DIR      = os.path.join(BASE_DIR, "results", "models")
CACHE_DIR       = os.path.join(BASE_DIR, "results", "cache")
DOCS_DIR        = os.path.join(BASE_DIR, "docs")
SCRIPTS_DIR     = os.path.join(BASE_DIR, "scripts")

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Development Mode ───────────────────────────────────────────────────
# True  = use small sample for fast iteration during development
# False = use full dataset for final paper results
DEV_MODE        = False
DEV_SAMPLE_SIZE = 500

# ── Cheap Mode ────────────────────────────────────────────────────────
# Use this for local development on small hardware.
# It reduces evaluation cost without changing the core architecture.
CHEAP_MODE = _env_flag("CHEAP_MODE", False)
CHEAP_EVAL_SAMPLE_SIZE = int(os.getenv("CHEAP_EVAL_SAMPLE_SIZE", "100"))
CHEAP_ABLATION_SAMPLE_SIZE = int(os.getenv("CHEAP_ABLATION_SAMPLE_SIZE", "50"))
CHEAP_MAX_SUBCLAIMS = int(os.getenv("CHEAP_MAX_SUBCLAIMS", "2"))
ENSEMBLE_SAMPLE_SIZE = int(os.getenv("ENSEMBLE_SAMPLE_SIZE", "0"))
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "0"))
ABLATION_SAMPLE_SIZE = int(os.getenv("ABLATION_SAMPLE_SIZE", "0"))

# ── Dataset Settings ───────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# LIAR column names — taken directly from original dataset script
# Source: https://huggingface.co/datasets/ucsbnlp/liar
LIAR_COLUMNS = [
    "id", "label", "statement", "subject",
    "speaker", "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

# 6-class LIAR labels mapped to binary
# true/mostly-true/half-true  → REAL (1)
# barely-true/false/pants-fire → FAKE (0)
# Why binary? Research question is fake vs real.
# Makes comparison with other binary classifiers fair.
LABEL_MAP = {
    "true"        : 1,
    "mostly-true" : 1,
    "half-true"   : 1,
    "barely-true" : 0,
    "false"       : 0,
    "pants-fire"  : 0
}
LABEL_NAMES = ["FAKE", "REAL"]

# ── BERT / RoBERTa Settings ────────────────────────────────────────────
# Why roberta-base and not bert-base-uncased?
# RoBERTa removed Next Sentence Prediction, trained on 10x more data.
# Consistently 2-4% better on classification benchmarks.
BERT_MODEL_NAME = "roberta-base"
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "512"))
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "16"))
BERT_EPOCHS     = int(os.getenv("BERT_EPOCHS", "3"))
BERT_LR         = float(os.getenv("BERT_LR", "2e-5"))

# ── Embedding Model Settings ───────────────────────────────────────────
# Why all-MiniLM-L6-v2?
# Designed for semantic sentence similarity.
# Fast inference, strong performance, 384-dim output.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── RAG Retrieval Settings ─────────────────────────────────────────────
TOP_K_RETRIEVAL = 5
CHUNK_SIZE      = 300
CHUNK_OVERLAP   = 50

# ── Source Credibility Weights ─────────────────────────────────────────
SOURCE_WEIGHTS = {
    "reuters"    : 0.90,
    "who"        : 0.95,
    "politifact" : 0.85,
    "wikipedia"  : 0.70,
    "unknown"    : 0.30
}

# ── Temporal Weighting ─────────────────────────────────────────────────
MAX_AGE_DAYS   = 365 * 2
RECENCY_WEIGHT = 0.20

# ── Hallucination Detection Thresholds ────────────────────────────────
HALLUCINATION_SIM_THRESHOLD  = 0.45
HALLUCINATION_CONF_THRESHOLD = 0.75

# ── LLM Settings ──────────────────────────────────────────────────────
# Temperature 0.0 = deterministic = reproducible research results
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL       = os.getenv(
    "LLM_MODEL",
    (
        "gemini-2.5-flash" if LLM_PROVIDER.lower() == "gemini"
        else "qwen2.5:7b" if LLM_PROVIDER.lower() == "ollama"
        else "llama-3.3-70b-versatile"
    )
)
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 1000
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))
LLM_RATE_LIMIT_RPM = int(os.getenv("LLM_RATE_LIMIT_RPM", "30"))
LLM_MAX_RETRIES    = int(os.getenv("LLM_MAX_RETRIES", "6"))
LLM_RETRY_BASE_SEC = float(os.getenv("LLM_RETRY_BASE_SEC", "2.0"))

# ── Dataset Selection ──────────────────────────────────────────────────
# Switch between LIAR and ISOT training
# Used for cross-dataset evaluation in paper
DATASET = _normalize_dataset_name(os.getenv("DATASET", "isot"))

# ── Benchmark Profile ──────────────────────────────────────────────────
# Keep the benchmark intent explicit:
# - verification          : external-evidence claim verification
# - article_classification: news/article classification benchmark
BENCHMARK_PROFILE = os.getenv(
    "BENCHMARK_PROFILE",
    "verification" if DATASET == "liar" else "article_classification"
).strip().lower()

# ── Review / Reporting Safety ──────────────────────────────────────────
# These flags do not change the model architecture; they control how
# strict the reporting layer is about caveats, saved outputs, and full
# versus development-style runs.
STRICT_REVIEW_MODE = _env_flag("STRICT_REVIEW_MODE", False)
SAVE_ALL_EVAL_RESULTS = _env_flag("SAVE_ALL_EVAL_RESULTS", False)
REVIEW_AUDIT_PATH = os.path.join(TABLES_DIR, f"reviewer_audit_{DATASET}.json")


def get_benchmark_profile(dataset=None):
    """Returns the benchmark profile for a dataset/configuration."""

    dataset = _normalize_dataset_name(dataset) or DATASET
    default_profile = (
        "verification" if dataset == "liar" else "article_classification"
    )
    return os.getenv("BENCHMARK_PROFILE", default_profile).strip().lower()


def get_benchmark_notes(dataset=None, benchmark_profile=None):
    """
    Returns benchmark caveats used by evaluation/reporting scripts.

    The goal is not to block experimentation, but to keep the saved
    artifacts honest about what a benchmark does and does not prove.
    """

    dataset = _normalize_dataset_name(dataset) or DATASET
    benchmark_profile = (
        benchmark_profile or get_benchmark_profile(dataset)
    ).strip().lower()

    notes = []
    severe_issues = []

    if dataset == "isot":
        notes.append(
            "ISOT is an article-classification benchmark whose real/fake "
            "labels are highly correlated with publication/source style "
            "(Reuters vs non-Reuters content)."
        )
        if benchmark_profile == "verification":
            severe_issues.append(
                "ISOT should not be treated as the primary external-evidence "
                "fact-verification benchmark."
            )

    if dataset == "liar":
        notes.append(
            "LIAR contains short human claims and is better aligned with "
            "claim-level verification than ISOT."
        )

    if CHEAP_MODE:
        notes.append(
            "Cheap mode is enabled; sampled evaluations are suitable for "
            "development, not headline final-paper claims."
        )

    return {
        "dataset": dataset,
        "benchmark_profile": benchmark_profile,
        "notes": notes,
        "severe_issues": severe_issues,
    }


def get_processed_split_path(split, dataset=None):
    """Returns the processed CSV path for a dataset split."""

    dataset = _normalize_dataset_name(dataset)
    filename = f"{split}.csv" if dataset == "liar" else f"{dataset}_{split}.csv"
    return os.path.join(DATA_PROCESSED, filename)


def get_roberta_model_dir(dataset=None):
    """Returns the preferred RoBERTa checkpoint directory for a dataset."""

    dataset = _normalize_dataset_name(dataset)
    preferred = os.path.join(MODELS_DIR, f"roberta_{dataset}")
    if os.path.isdir(preferred):
        return preferred
    return os.path.join(MODELS_DIR, "roberta_liar")


def get_ensemble_path(dataset=None):
    """
    Returns the ensemble path for a dataset.

    Dataset-specific ensembles are preferred. The legacy shared ensemble is
    still accepted as a fallback so existing experiments keep working.
    """

    dataset = _normalize_dataset_name(dataset)
    preferred = os.path.join(MODELS_DIR, f"ensemble_{dataset}.pkl")
    if os.path.exists(preferred):
        return preferred

    legacy = os.path.join(MODELS_DIR, "ensemble.pkl")
    if os.path.exists(legacy):
        return legacy

    return preferred


def get_preferred_ensemble_path(dataset=None):
    """Returns the dataset-specific ensemble path without legacy fallback."""

    dataset = _normalize_dataset_name(dataset)
    return os.path.join(MODELS_DIR, f"ensemble_{dataset}.pkl")


def get_knowledge_base_dir(dataset=None):
    """
    Returns the knowledge-base directory for a dataset.

    Dataset-specific stores live under ``data/knowledge_base/<dataset>``.
    The old shared directory remains a fallback for backwards compatibility.
    """

    dataset = _normalize_dataset_name(dataset)
    preferred = os.path.join(KNOWLEDGE_BASE, dataset)
    if os.path.isdir(preferred):
        return preferred
    return KNOWLEDGE_BASE

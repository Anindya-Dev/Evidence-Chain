# config.py
# EvidenceChain — Central Configuration
# All settings live here. Change here, changes everywhere.
# This ensures reproducibility — a core requirement of research.

import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
KNOWLEDGE_BASE  = os.path.join(BASE_DIR, "data", "knowledge_base")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(BASE_DIR, "results", "plots")
TABLES_DIR      = os.path.join(BASE_DIR, "results", "tables")
MODELS_DIR      = os.path.join(BASE_DIR, "results", "models")

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Development Mode ───────────────────────────────────────────────────
# True  = use small sample for fast iteration during development
# False = use full dataset for final paper results
DEV_MODE        = False
DEV_SAMPLE_SIZE = 500

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
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 16
BERT_EPOCHS     = 3
BERT_LR         = 2e-5

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
# Groq provides free API access to llama3
# Temperature 0.0 = deterministic = reproducible research results
LLM_PROVIDER    = "groq"
LLM_MODEL       = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 1000
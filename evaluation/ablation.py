# evaluation/ablation.py
# Ablation study — removes one component at a time
# and measures the performance drop.
#
# Why ablation study?
# Proves every architectural choice is justified.
# Reviewers ask: "Is each component actually necessary?"
# Ablation answers: "Yes — here is the proof."

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.preprocessor import build_model_input, extract_claim_text
from modules.retriever     import Retriever
from modules.decomposer    import ClaimDecomposer
from modules.reasoner      import EvidenceReasoner
from modules.ensemble      import StackingEnsemble


def _resolve_model_dir():
    """Returns the best available RoBERTa checkpoint for the active dataset."""

    return config.get_roberta_model_dir()


def _resolve_test_data_path():
    """Returns the processed test split for the active dataset."""

    return config.get_processed_split_path("test")


def _resolve_text_column(df):
    """
    Returns the input column used for ablation inference.

    LIAR uses raw claims in ``statement``. ISOT uses ``combined_text``
    because its processed split does not contain a claim column.
    """

    if "statement" in df.columns:
        return "statement"
    if "combined_text" in df.columns:
        return "combined_text"

    raise KeyError(
        "Could not find an ablation text column. "
        "Expected 'statement' or 'combined_text'."
    )


def _sample_evaluation_df(df, sample_size):
    """Returns a reproducible stratified sample for ablation runs."""

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


def _limit_sub_claims(sub_claims):
    """Applies the shared cheap-mode sub-claim cap."""

    if config.CHEAP_MODE and len(sub_claims) > config.CHEAP_MAX_SUBCLAIMS:
        return sub_claims[:config.CHEAP_MAX_SUBCLAIMS]
    return sub_claims


def _success(pred, hallucinated):
    """Standard ablation return payload for a successful configuration run."""

    return {
        "pred": int(pred),
        "hallucinated": hallucinated,
        "failed": False,
        "error": None,
    }


def _failure(error):
    """Standard ablation return payload for failed configuration runs."""

    return {
        "pred": None,
        "hallucinated": None,
        "failed": True,
        "error": str(error),
    }


# ── Helper: Get BERT probability ──────────────────────────────────────
def get_bert_prob(text, model, tokenizer, device):
    """Returns BERT probability of REAL class."""
    model_input = build_model_input(text)
    encoding = tokenizer(
        model_input,
        max_length     = config.BERT_MAX_LENGTH,
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids      = encoding["input_ids"].to(device),
            attention_mask = encoding["attention_mask"].to(device)
        )
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()


# ── Configuration 1: Full EvidenceChain ───────────────────────────────
def run_full_pipeline(example, retriever, decomposer, reasoner,
                      ensemble, bert_model, tokenizer, device):
    """Runs the complete EvidenceChain pipeline."""
    try:
        claim = extract_claim_text(example)

        # Decompose
        sub_claims = _limit_sub_claims(decomposer.decompose(claim))

        # Retrieve + reason per sub-claim
        sub_results = []
        for sc in sub_claims:
            evidence = retriever.retrieve(sc)
            result   = reasoner.reason(sc, evidence)
            result["source_weight_avg"] = round(
                sum(e["source_weight"] for e in evidence) / len(evidence), 4
            ) if evidence else 0.3
            result["max_similarity"] = max(
                (e["similarity"] for e in evidence), default=0.0
            )
            sub_results.append(result)

        # Aggregate
        rag_result = reasoner.aggregate_sub_claim_verdicts(sub_results)
        rag_result["max_similarity"]    = round(
            sum(r["max_similarity"] for r in sub_results) / len(sub_results), 4
        )
        rag_result["source_weight_avg"] = round(
            sum(r["source_weight_avg"] for r in sub_results) / len(sub_results), 4
        )

        # BERT
        bert_prob = get_bert_prob(example, bert_model, tokenizer, device)

        # Ensemble
        final = ensemble.predict_single(bert_prob, rag_result)
        pred  = 1 if final["final_verdict"] == "REAL" else 0

        return _success(pred, rag_result.get("hallucination_flag", False))

    except Exception as e:
        return _failure(e)


# ── Configuration 2: No Claim Decomposition ───────────────────────────
def run_no_decomposition(example, retriever, reasoner,
                         ensemble, bert_model, tokenizer, device):
    """
    Removes claim decomposition — treats whole claim as one unit.
    This tests C1 contribution.
    """
    try:
        claim = extract_claim_text(example)

        # No decomposition — treat whole claim as one sub-claim
        evidence = retriever.retrieve(claim)
        result   = reasoner.reason(claim, evidence)
        result["source_weight_avg"] = round(
            sum(e["source_weight"] for e in evidence) / len(evidence), 4
        ) if evidence else 0.3
        result["max_similarity"] = max(
            (e["similarity"] for e in evidence), default=0.0
        )

        rag_result = {
            "final_verdict"      : result["verdict"],
            "final_confidence"   : result["confidence"],
            "hallucination_flag" : result["hallucination_flag"],
            "sub_claim_count"    : 1,
            "max_similarity"     : result["max_similarity"],
            "source_weight_avg"  : result["source_weight_avg"]
        }

        bert_prob = get_bert_prob(example, bert_model, tokenizer, device)
        final     = ensemble.predict_single(bert_prob, rag_result)
        pred      = 1 if final["final_verdict"] == "REAL" else 0

        return _success(pred, result["hallucination_flag"])

    except Exception as e:
        return _failure(e)


# ── Configuration 3: No Temporal + Source Weighting ───────────────────
def run_no_weighting(example, retriever, decomposer, reasoner,
                     ensemble, bert_model, tokenizer, device):
    """
    Removes credibility and temporal weighting — all sources equal weight.
    This tests C2 contribution.
    """
    try:
        claim = extract_claim_text(example)
        sub_claims  = _limit_sub_claims(decomposer.decompose(claim))
        sub_results = []

        for sc in sub_claims:
            evidence = retriever.retrieve(sc, use_weighting=False)

            # Override prompt-visible weights as well so the LLM does not
            # still see the original credibility values during reasoning.
            for e in evidence:
                e["source_weight"]  = 0.5   # neutral weight
                e["recency_weight"] = 0.0   # no recency bonus

            result = reasoner.reason(sc, evidence)
            result["source_weight_avg"] = 0.5
            result["max_similarity"]    = max(
                (e["similarity"] for e in evidence), default=0.0
            )
            sub_results.append(result)

        rag_result = reasoner.aggregate_sub_claim_verdicts(sub_results)
        rag_result["max_similarity"]    = round(
            sum(r["max_similarity"] for r in sub_results) / len(sub_results), 4
        )
        rag_result["source_weight_avg"] = 0.5

        bert_prob = get_bert_prob(example, bert_model, tokenizer, device)
        final     = ensemble.predict_single(bert_prob, rag_result)
        pred      = 1 if final["final_verdict"] == "REAL" else 0

        return _success(pred, rag_result.get("hallucination_flag", False))

    except Exception as e:
        return _failure(e)


# ── Configuration 4: No Hallucination Detection ───────────────────────
def run_no_hallucination(example, retriever, decomposer, reasoner,
                         ensemble, bert_model, tokenizer, device):
    """
    Disables hallucination detection — never flags any verdict.
    This tests C3 contribution — measures HR increase.
    """
    try:
        claim = extract_claim_text(example)
        sub_claims  = _limit_sub_claims(decomposer.decompose(claim))
        sub_results = []

        for sc in sub_claims:
            evidence = retriever.retrieve(sc)
            result   = reasoner.reason(sc, evidence)

            # Override hallucination flag — always False
            result["hallucination_flag"] = False
            result["source_weight_avg"]  = round(
                sum(e["source_weight"] for e in evidence) / len(evidence), 4
            ) if evidence else 0.3
            result["max_similarity"] = max(
                (e["similarity"] for e in evidence), default=0.0
            )
            sub_results.append(result)

        rag_result = reasoner.aggregate_sub_claim_verdicts(sub_results)
        rag_result["max_similarity"]    = round(
            sum(r["max_similarity"] for r in sub_results) / len(sub_results), 4
        )
        rag_result["source_weight_avg"] = round(
            sum(r["source_weight_avg"] for r in sub_results) / len(sub_results), 4
        )
        rag_result["hallucination_flag"] = False

        bert_prob = get_bert_prob(example, bert_model, tokenizer, device)
        final     = ensemble.predict_single(bert_prob, rag_result)
        pred      = 1 if final["final_verdict"] == "REAL" else 0

        return _success(pred, None)  # hallucination metric not applicable

    except Exception as e:
        return _failure(e)


# ── Configuration 5: BERT Only ────────────────────────────────────────
def run_bert_only(example, bert_model, tokenizer, device):
    """
    BERT classifier only — no RAG at all.
    Shows what system achieves without evidence retrieval.
    """
    try:
        bert_prob = get_bert_prob(example, bert_model, tokenizer, device)
        pred      = 1 if bert_prob > 0.5 else 0
        return _success(pred, None)
    except Exception as e:
        return _failure(e)


# ── Configuration 6: RAG Only ─────────────────────────────────────────
def run_rag_only(example, retriever, decomposer, reasoner):
    """
    RAG pipeline only — no BERT classifier.
    Shows what system achieves without linguistic analysis.
    """
    try:
        claim = extract_claim_text(example)
        sub_claims  = _limit_sub_claims(decomposer.decompose(claim))
        sub_results = []

        for sc in sub_claims:
            evidence = retriever.retrieve(sc)
            result   = reasoner.reason(sc, evidence)
            result["source_weight_avg"] = round(
                sum(e["source_weight"] for e in evidence) / len(evidence), 4
            ) if evidence else 0.3
            result["max_similarity"] = max(
                (e["similarity"] for e in evidence), default=0.0
            )
            sub_results.append(result)

        rag_result = reasoner.aggregate_sub_claim_verdicts(sub_results)

        # Convert RAG verdict to binary
        pred = 1 if rag_result["final_verdict"] == "TRUE" else 0
        return _success(pred, rag_result.get("hallucination_flag", False))

    except Exception as e:
        return _failure(e)


# ── Run Ablation ──────────────────────────────────────────────────────
def run_ablation(sample_size=30):
    """
    Runs full ablation study on the active dataset test set.
    
    Args:
        sample_size : number of test claims to evaluate
                      30 is enough for ablation — we want
                      relative differences not absolute numbers
    """

    if sample_size == 30 and config.CHEAP_MODE:
        sample_size = config.CHEAP_ABLATION_SAMPLE_SIZE

    print("="*60)
    print("  EVIDENCECHAIN — ABLATION STUDY")
    print("="*60)

    # Load components
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Dataset: {config.DATASET}")
    if config.CHEAP_MODE:
        print(
            f"Cheap mode enabled: {sample_size} ablation examples, "
            f"max {config.CHEAP_MAX_SUBCLAIMS} sub-claims"
        )

    retriever  = Retriever()
    decomposer = ClaimDecomposer()
    reasoner   = EvidenceReasoner()
    ensemble   = StackingEnsemble()
    ensemble.load()

    model_dir = _resolve_model_dir()
    tokenizer  = RobertaTokenizer.from_pretrained(model_dir)
    bert_model = RobertaForSequenceClassification.from_pretrained(model_dir)
    bert_model.to(device)
    bert_model.eval()

    # Load test data
    test_df = pd.read_csv(_resolve_test_data_path())
    text_column = _resolve_text_column(test_df)
    test_df = test_df.dropna(subset=["binary_label", text_column])
    test_df = _sample_evaluation_df(test_df, sample_size)

    print(f"\nEvaluating on {sample_size} test claims...")

    # Store results for each configuration
    configs = {
        "Full EvidenceChain"        : {"preds": [], "hallucinated": [], "failures": 0},
        "No Decomposition"          : {"preds": [], "hallucinated": [], "failures": 0},
        "No Weighting (C2)"         : {"preds": [], "hallucinated": [], "failures": 0},
        "No Hallucination Det. (C3)": {"preds": [], "hallucinated": [], "failures": 0},
        "BERT Only"                 : {"preds": [], "hallucinated": [], "failures": 0},
        "RAG Only"                  : {"preds": [], "hallucinated": [], "failures": 0},
    }
    true_labels = []

    for i, row in test_df.iterrows():
        example = row
        label = int(row["binary_label"])
        true_labels.append(label)

        if (len(true_labels)) % 5 == 0:
            print(f"  Progress: {len(true_labels)}/{sample_size}")

        # Run all configurations
        result = run_full_pipeline(
            example, retriever, decomposer, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        if result["failed"]:
            configs["Full EvidenceChain"]["failures"] += 1
            result["pred"] = 1 - label
        configs["Full EvidenceChain"]["preds"].append(result["pred"])
        if result["hallucinated"] is not None:
            configs["Full EvidenceChain"]["hallucinated"].append(
                result["hallucinated"]
            )

        result = run_no_decomposition(
            example, retriever, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        if result["failed"]:
            configs["No Decomposition"]["failures"] += 1
            result["pred"] = 1 - label
        configs["No Decomposition"]["preds"].append(result["pred"])
        if result["hallucinated"] is not None:
            configs["No Decomposition"]["hallucinated"].append(
                result["hallucinated"]
            )

        result = run_no_weighting(
            example, retriever, decomposer, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        if result["failed"]:
            configs["No Weighting (C2)"]["failures"] += 1
            result["pred"] = 1 - label
        configs["No Weighting (C2)"]["preds"].append(result["pred"])
        if result["hallucinated"] is not None:
            configs["No Weighting (C2)"]["hallucinated"].append(
                result["hallucinated"]
            )

        result = run_no_hallucination(
            example, retriever, decomposer, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        if result["failed"]:
            configs["No Hallucination Det. (C3)"]["failures"] += 1
            result["pred"] = 1 - label
        configs["No Hallucination Det. (C3)"]["preds"].append(result["pred"])
        if result["hallucinated"] is not None:
            configs["No Hallucination Det. (C3)"]["hallucinated"].append(
                result["hallucinated"]
            )

        result = run_bert_only(example, bert_model, tokenizer, device)
        if result["failed"]:
            configs["BERT Only"]["failures"] += 1
            result["pred"] = 1 - label
        configs["BERT Only"]["preds"].append(result["pred"])
        if result["hallucinated"] is not None:
            configs["BERT Only"]["hallucinated"].append(result["hallucinated"])

        result = run_rag_only(example, retriever, decomposer, reasoner)
        if result["failed"]:
            configs["RAG Only"]["failures"] += 1
            result["pred"] = 1 - label
        configs["RAG Only"]["preds"].append(result["pred"])
        if result["hallucinated"] is not None:
            configs["RAG Only"]["hallucinated"].append(result["hallucinated"])

    # Compute metrics for each configuration
    print("\n\n" + "="*65)
    print("  ABLATION RESULTS")
    print("="*65)
    print("  Note: fusion uses the saved ensemble as a fixed meta-classifier.")
    print(
        f"{'Configuration':<30} {'Accuracy':>10} {'F1':>8} {'HR':>8} {'Fail':>8}"
    )
    print("-"*65)

    results = []
    for config_name, data in configs.items():
        acc = round(accuracy_score(true_labels, data["preds"]), 4)
        f1  = round(f1_score(true_labels, data["preds"],
                             average="weighted", zero_division=0), 4)
        if data["hallucinated"]:
            hr = round(
                sum(data["hallucinated"]) / len(data["hallucinated"]), 4
            )
        else:
            hr = "N/A"

        print(
            f"{config_name:<30} {acc:>10.4f} {f1:>8.4f} "
            f"{str(hr):>8} {data['failures']:>8}"
        )
        results.append({
            "configuration"       : config_name,
            "accuracy"            : acc,
            "f1"                  : f1,
            "hallucination_rate"  : hr,
            "failure_count"       : data["failures"],
            "sample_size"         : sample_size,
            "cheap_mode"          : config.CHEAP_MODE,
            "fusion_strategy"     : "fixed_saved_ensemble",
        })

    print("="*65)

    # Save results
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(config.TABLES_DIR, f"ablation_results_{config.DATASET}.csv"),
        index=False
    )
    print(
        f"\nAblation results saved to "
        f"results/tables/ablation_results_{config.DATASET}.csv"
    )

    return results_df


if __name__ == "__main__":
    # 30 claims is enough for ablation
    # We want relative differences between configs
    # not absolute performance numbers
    results = run_ablation(sample_size=30)

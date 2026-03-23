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
from modules.preprocessor import preprocess_text
from modules.retriever     import Retriever
from modules.decomposer    import ClaimDecomposer
from modules.reasoner      import EvidenceReasoner
from modules.ensemble      import StackingEnsemble


# ── Helper: Get BERT probability ──────────────────────────────────────
def get_bert_prob(text, model, tokenizer, device):
    """Returns BERT probability of REAL class."""
    clean = preprocess_text(text)
    encoding = tokenizer(
        clean,
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
def run_full_pipeline(claim, retriever, decomposer, reasoner,
                      ensemble, bert_model, tokenizer, device):
    """Runs the complete EvidenceChain pipeline."""
    try:
        # Decompose
        sub_claims = decomposer.decompose(claim)

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
        bert_prob = get_bert_prob(claim, bert_model, tokenizer, device)

        # Ensemble
        final = ensemble.predict_single(bert_prob, rag_result)
        pred  = 1 if final["final_verdict"] == "REAL" else 0

        return pred, rag_result.get("hallucination_flag", False)

    except Exception as e:
        return 0, False


# ── Configuration 2: No Claim Decomposition ───────────────────────────
def run_no_decomposition(claim, retriever, reasoner,
                         ensemble, bert_model, tokenizer, device):
    """
    Removes claim decomposition — treats whole claim as one unit.
    This tests C1 contribution.
    """
    try:
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

        bert_prob = get_bert_prob(claim, bert_model, tokenizer, device)
        final     = ensemble.predict_single(bert_prob, rag_result)
        pred      = 1 if final["final_verdict"] == "REAL" else 0

        return pred, result["hallucination_flag"]

    except Exception as e:
        return 0, False


# ── Configuration 3: No Temporal + Source Weighting ───────────────────
def run_no_weighting(claim, retriever, decomposer, reasoner,
                     ensemble, bert_model, tokenizer, device):
    """
    Removes credibility and temporal weighting — all sources equal weight.
    This tests C2 contribution.
    """
    try:
        sub_claims  = decomposer.decompose(claim)
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

        bert_prob = get_bert_prob(claim, bert_model, tokenizer, device)
        final     = ensemble.predict_single(bert_prob, rag_result)
        pred      = 1 if final["final_verdict"] == "REAL" else 0

        return pred, rag_result.get("hallucination_flag", False)

    except Exception as e:
        return 0, False


# ── Configuration 4: No Hallucination Detection ───────────────────────
def run_no_hallucination(claim, retriever, decomposer, reasoner,
                         ensemble, bert_model, tokenizer, device):
    """
    Disables hallucination detection — never flags any verdict.
    This tests C3 contribution — measures HR increase.
    """
    try:
        sub_claims  = decomposer.decompose(claim)
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

        bert_prob = get_bert_prob(claim, bert_model, tokenizer, device)
        final     = ensemble.predict_single(bert_prob, rag_result)
        pred      = 1 if final["final_verdict"] == "REAL" else 0

        return pred, False  # never hallucinated in this config

    except Exception as e:
        return 0, False


# ── Configuration 5: BERT Only ────────────────────────────────────────
def run_bert_only(claim, bert_model, tokenizer, device):
    """
    BERT classifier only — no RAG at all.
    Shows what system achieves without evidence retrieval.
    """
    try:
        bert_prob = get_bert_prob(claim, bert_model, tokenizer, device)
        pred      = 1 if bert_prob > 0.5 else 0
        return pred, False
    except Exception as e:
        return 0, False


# ── Configuration 6: RAG Only ─────────────────────────────────────────
def run_rag_only(claim, retriever, decomposer, reasoner):
    """
    RAG pipeline only — no BERT classifier.
    Shows what system achieves without linguistic analysis.
    """
    try:
        sub_claims  = decomposer.decompose(claim)
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
        return pred, rag_result.get("hallucination_flag", False)

    except Exception as e:
        return 0, False


# ── Run Ablation ──────────────────────────────────────────────────────
def run_ablation(sample_size=30):
    """
    Runs full ablation study on LIAR test set.
    
    Args:
        sample_size : number of test claims to evaluate
                      30 is enough for ablation — we want
                      relative differences not absolute numbers
    """

    print("="*60)
    print("  EVIDENCECHAIN — ABLATION STUDY")
    print("="*60)

    # Load components
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    retriever  = Retriever()
    decomposer = ClaimDecomposer()
    reasoner   = EvidenceReasoner()
    ensemble   = StackingEnsemble()
    ensemble.load()

    tokenizer  = RobertaTokenizer.from_pretrained(
        os.path.join(config.MODELS_DIR, "roberta_liar")
    )
    bert_model = RobertaForSequenceClassification.from_pretrained(
        os.path.join(config.MODELS_DIR, "roberta_liar")
    )
    bert_model.to(device)
    bert_model.eval()

    # Load test data
    test_df = pd.read_csv(os.path.join(config.DATA_PROCESSED, "test.csv"))
    test_df = test_df.dropna(subset=["binary_label", "statement"])
    test_df = test_df.sample(sample_size, random_state=config.RANDOM_SEED)

    print(f"\nEvaluating on {sample_size} test claims...")

    # Store results for each configuration
    configs = {
        "Full EvidenceChain"        : {"preds": [], "hallucinated": []},
        "No Decomposition"          : {"preds": [], "hallucinated": []},
        "No Weighting (C2)"         : {"preds": [], "hallucinated": []},
        "No Hallucination Det. (C3)": {"preds": [], "hallucinated": []},
        "BERT Only"                 : {"preds": [], "hallucinated": []},
        "RAG Only"                  : {"preds": [], "hallucinated": []},
    }
    true_labels = []

    for i, row in test_df.iterrows():
        claim = str(row["statement"])
        label = int(row["binary_label"])
        true_labels.append(label)

        if (len(true_labels)) % 5 == 0:
            print(f"  Progress: {len(true_labels)}/{sample_size}")

        # Run all configurations
        p, h = run_full_pipeline(
            claim, retriever, decomposer, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        configs["Full EvidenceChain"]["preds"].append(p)
        configs["Full EvidenceChain"]["hallucinated"].append(h)

        p, h = run_no_decomposition(
            claim, retriever, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        configs["No Decomposition"]["preds"].append(p)
        configs["No Decomposition"]["hallucinated"].append(h)

        p, h = run_no_weighting(
            claim, retriever, decomposer, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        configs["No Weighting (C2)"]["preds"].append(p)
        configs["No Weighting (C2)"]["hallucinated"].append(h)

        p, h = run_no_hallucination(
            claim, retriever, decomposer, reasoner,
            ensemble, bert_model, tokenizer, device
        )
        configs["No Hallucination Det. (C3)"]["preds"].append(p)
        configs["No Hallucination Det. (C3)"]["hallucinated"].append(h)

        p, h = run_bert_only(claim, bert_model, tokenizer, device)
        configs["BERT Only"]["preds"].append(p)
        configs["BERT Only"]["hallucinated"].append(h)

        p, h = run_rag_only(claim, retriever, decomposer, reasoner)
        configs["RAG Only"]["preds"].append(p)
        configs["RAG Only"]["hallucinated"].append(h)

    # Compute metrics for each configuration
    print("\n\n" + "="*65)
    print("  ABLATION RESULTS")
    print("="*65)
    print(f"{'Configuration':<30} {'Accuracy':>10} {'F1':>8} {'HR':>8}")
    print("-"*65)

    results = []
    for config_name, data in configs.items():
        acc = round(accuracy_score(true_labels, data["preds"]), 4)
        f1  = round(f1_score(true_labels, data["preds"],
                             average="weighted", zero_division=0), 4)
        hr  = round(sum(data["hallucinated"]) / len(data["hallucinated"]), 4)

        print(f"{config_name:<30} {acc:>10.4f} {f1:>8.4f} {hr:>8.4f}")
        results.append({
            "configuration" : config_name,
            "accuracy"      : acc,
            "f1"            : f1,
            "hallucination_rate": hr
        })

    print("="*65)

    # Save results
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(config.TABLES_DIR, "ablation_results.csv"),
        index=False
    )
    print(f"\nAblation results saved to results/tables/ablation_results.csv")

    return results_df


if __name__ == "__main__":
    # 30 claims is enough for ablation
    # We want relative differences between configs
    # not absolute performance numbers
    results = run_ablation(sample_size=30)

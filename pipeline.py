# pipeline.py
# EvidenceChain — Full Pipeline
# Connects every module end to end.
# One claim in → full verdict out.

import os
import sys
import json
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.preprocessor import preprocess_text
from modules.decomposer   import ClaimDecomposer
from modules.retriever    import Retriever
from modules.reasoner     import EvidenceReasoner
from modules.ensemble     import StackingEnsemble
import config


class EvidenceChain:
    """
    Full EvidenceChain pipeline.
    Loads all modules once, processes claims efficiently.
    """

    def __init__(self):
        print("Initializing EvidenceChain...")
        print("-" * 50)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Device: {self.device}")

        # Load all modules
        self.decomposer = ClaimDecomposer()
        self.retriever  = Retriever()
        self.reasoner   = EvidenceReasoner()
        self.ensemble   = StackingEnsemble()

        # Load trained BERT model
        print("Loading RoBERTa model...")
        bert_path       = os.path.join(config.MODELS_DIR, "roberta_liar")
        self.tokenizer  = RobertaTokenizer.from_pretrained(bert_path)
        self.bert_model = RobertaForSequenceClassification.from_pretrained(
            bert_path
        )
        self.bert_model.to(self.device)
        self.bert_model.eval()

        # Load trained ensemble
        self.ensemble.load()

        print("-" * 50)
        print("EvidenceChain ready\n")

    def get_bert_probability(self, claim):
        """
        Gets BERT probability of REAL class for a claim.
        
        Returns:
            float — probability of REAL (0 to 1)
        """

        # Preprocess
        clean = preprocess_text(claim)

        # Tokenize
        encoding = self.tokenizer(
            clean,
            max_length     = config.BERT_MAX_LENGTH,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass — no gradient needed
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )

        # Softmax to get probabilities
        probs     = torch.softmax(outputs.logits, dim=1)
        real_prob = probs[0][1].item()  # index 1 = REAL class

        return real_prob

    def verify(self, claim):
        """
        Full verification pipeline for a single claim.
        
        Args:
            claim : raw claim string
            
        Returns:
            dict with complete verdict and explanation
        """

        print(f"\n{'='*60}")
        print(f"CLAIM: {claim}")
        print(f"{'='*60}")

        # ── Step 1: Preprocess ─────────────────────────────────
        clean_claim = preprocess_text(claim)
        print(f"\n[1] Preprocessed: {clean_claim}")

        # ── Step 2: Decompose ──────────────────────────────────
        print(f"\n[2] Decomposing claim...")
        sub_claims = self.decomposer.decompose(clean_claim)
        print(f"    Sub-claims ({len(sub_claims)}):")
        for i, sc in enumerate(sub_claims):
            print(f"    {i+1}. {sc}")

        # ── Step 3: Retrieve + Reason per sub-claim ────────────
        print(f"\n[3] Retrieving evidence and reasoning...")
        sub_claim_results = []

        for sc in sub_claims:
            evidence = self.retriever.retrieve(sc, top_k=config.TOP_K_RETRIEVAL)
            result   = self.reasoner.reason(sc, evidence)

            # Add source weight average for ensemble feature
            if evidence:
                result["source_weight_avg"] = round(
                    sum(e["source_weight"] for e in evidence) / len(evidence), 4
                )
                result["max_similarity"] = max(
                    e["similarity"] for e in evidence
                )
            else:
                result["source_weight_avg"] = 0.3
                result["max_similarity"]    = 0.0

            sub_claim_results.append(result)

            print(f"\n    Sub-claim : {sc}")
            print(f"    Verdict   : {result['verdict']}")
            print(f"    Confidence: {result['confidence']}")
            print(f"    Reasoning : {result['reasoning']}")

        # ── Step 4: Aggregate sub-claim verdicts ───────────────
        rag_result = self.reasoner.aggregate_sub_claim_verdicts(
            sub_claim_results
        )

        # Add averaged features for ensemble
        rag_result["max_similarity"] = round(
            sum(r["max_similarity"] for r in sub_claim_results) /
            len(sub_claim_results), 4
        )
        rag_result["source_weight_avg"] = round(
            sum(r["source_weight_avg"] for r in sub_claim_results) /
            len(sub_claim_results), 4
        )

        print(f"\n[4] RAG Aggregated Verdict: {rag_result['final_verdict']}")
        print(f"    Confidence            : {rag_result['final_confidence']}")
        print(f"    Hallucinated          : {rag_result['hallucination_flag']}")

        # ── Step 5: BERT probability ───────────────────────────
        bert_prob = self.get_bert_probability(claim)
        print(f"\n[5] BERT probability (REAL): {bert_prob:.4f}")

        # ── Step 6: Stacking Ensemble ──────────────────────────
        final = self.ensemble.predict_single(bert_prob, rag_result)
        print(f"\n[6] FINAL VERDICT: {final['final_verdict']}")
        print(f"    Confidence  : {final['confidence']}")

        # ── Full Output ────────────────────────────────────────
        output = {
            "claim"          : claim,
            "sub_claims"     : sub_claims,
            "rag_verdict"    : rag_result["final_verdict"],
            "rag_confidence" : rag_result["final_confidence"],
            "bert_prob"      : round(bert_prob, 4),
            "final_verdict"  : final["final_verdict"],
            "confidence"     : final["confidence"],
            "hallucinated"   : rag_result["hallucination_flag"],
            "sub_results"    : [
                {
                    "sub_claim" : sub_claims[i],
                    "verdict"   : sub_claim_results[i]["verdict"],
                    "reasoning" : sub_claim_results[i]["reasoning"]
                }
                for i in range(len(sub_claims))
            ]
        }

        return output


if __name__ == "__main__":

    # Test full pipeline on sample claims
    pipeline = EvidenceChain()

    test_claims = [
        "COVID vaccines cause infertility and are banned in Europe",
        "The unemployment rate is the lowest it has been in 50 years",
        "In the case of a catastrophic event the CDC offices in Atlanta will self-destruct"
    ]

    all_results = []

    for claim in test_claims:
        result = pipeline.verify(claim)
        all_results.append(result)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "pipeline_test.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to results/pipeline_test.json")

    # Summary table
    print("\n" + "="*60)
    print("  PIPELINE SUMMARY")
    print("="*60)
    print(f"{'Claim':<45} {'RAG':<15} {'BERT':<8} {'FINAL'}")
    print("-"*60)
    for r in all_results:
        claim_short = r["claim"][:42] + "..."
        print(
            f"{claim_short:<45} "
            f"{r['rag_verdict']:<15} "
            f"{r['bert_prob']:<8.3f} "
            f"{r['final_verdict']}"
        )
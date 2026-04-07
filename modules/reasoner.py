# modules/reasoner.py
# LLM reasoning module — verdicts each sub-claim against evidence.
#
# Why LLM for reasoning and not a classifier?
# A classifier gives a label but no explanation.
# LLM reads evidence and produces:
#   - A verdict grounded in specific evidence
#   - A confidence score
#   - A human-readable reasoning chain
# This makes the system explainable — a core research requirement.

import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.llm_client import LLMClient

load_dotenv()


class EvidenceReasoner:
    """
    Uses LLM to reason over retrieved evidence and verdict sub-claims.
    
    Also performs hallucination detection:
    If LLM confidence is high but evidence similarity is low,
    the verdict is flagged as potentially hallucinated.
    """

    def __init__(self):
        self.client = LLMClient()

    def reason(self, sub_claim, evidence_list):
        """
        Reasons over evidence to produce a verdict for a sub-claim.
        
        Args:
            sub_claim     : atomic claim string
            evidence_list : list of evidence dicts from retriever
            
        Returns:
            dict with verdict, confidence, reasoning, hallucination_flag
        """

        # Format evidence for prompt
        evidence_text = self._format_evidence(evidence_list)

        # Best similarity score from retrieved evidence
        # Used for hallucination detection
        max_similarity = max(
    [e["similarity"] for e in evidence_list], default=0.0
)

        prompt = f"""You are a fact-checking expert. Your task is to verify a claim using only the provided evidence.

Claim: "{sub_claim}"

Evidence:
{evidence_text}

Instructions:
1. Read the evidence carefully
2. Determine if the claim is supported, contradicted, or not addressed
3. Base your verdict ONLY on the provided evidence
4. Do not use outside knowledge

Return ONLY a JSON object with this exact format:
{{
    "verdict": "TRUE" or "FALSE" or "UNVERIFIABLE",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<one sentence explaining your verdict based on evidence>"
}}

Rules:
- TRUE: evidence clearly supports the claim
- FALSE: evidence clearly contradicts the claim  
- UNVERIFIABLE: evidence does not address the claim
- confidence reflects how strongly the evidence supports your verdict"""

        try:
            raw    = self.client.generate_json_text(prompt)
            result = json.loads(raw)

            # Validate required fields
            verdict    = result.get("verdict", "UNVERIFIABLE")
            confidence = float(result.get("confidence", 0.5))
            reasoning  = result.get("reasoning", "No reasoning provided")

            # Normalize verdict
            if verdict not in ["TRUE", "FALSE", "UNVERIFIABLE"]:
                verdict = "UNVERIFIABLE"

            # ── Hallucination Detection ────────────────────────────
            # C3 Novel Contribution
            # Flag when LLM is confident but evidence is weak
            # This means LLM may be reasoning from parametric memory
            # not from retrieved evidence — that is hallucination
            hallucination_flag = self._check_hallucination(
                confidence, max_similarity
            )

            return {
                "verdict"            : verdict,
                "confidence"         : round(confidence, 4),
                "reasoning"          : reasoning,
                "max_similarity"     : round(max_similarity, 4),
                "hallucination_flag" : hallucination_flag
            }

        except json.JSONDecodeError:
            print(f"  Warning: JSON parse failed in reasoner")
            return self._fallback_result()

        except Exception as e:
            print(f"  Warning: Reasoning failed — {e}")
            return self._fallback_result()

    def _format_evidence(self, evidence_list):
        """
        Formats evidence list into readable prompt text.
        Each piece of evidence shows source and text.
        """
        if not evidence_list:
            return "No evidence available."

        lines = []
        for i, e in enumerate(evidence_list):
            text = self._truncate_evidence_text(e["text"])
            lines.append(
                f"[{i+1}] Source: {e['source'].upper()} "
                f"(credibility: {e['source_weight']})\n"
                f"    {text}"
            )
        return "\n\n".join(lines)

    def _truncate_evidence_text(self, text, max_words=120):
        """
        Keeps only the most useful evidence span in the prompt.

        Large Reuters documents make local LLM prompts slow and brittle.
        Truncating each retrieved snippet keeps grounding signal while
        making local reasoning much more reliable on commodity hardware.
        """

        words = str(text).split()
        return " ".join(words[:max_words])

    def _check_hallucination(self, confidence, similarity):
        """
        Detects potential hallucination.
        
        Hallucination = LLM is confident (>0.75) but
                        evidence similarity is weak (<0.45)
        
        This means the LLM verdict is not grounded
        in retrieved evidence — it is using its own
        parametric memory instead.
        
        Formula from config:
        HALLUCINATION_SIM_THRESHOLD  = 0.45
        HALLUCINATION_CONF_THRESHOLD = 0.75
        """
        return (
            similarity  < config.HALLUCINATION_SIM_THRESHOLD and
            confidence  > config.HALLUCINATION_CONF_THRESHOLD
        )

    def _fallback_result(self):
        """Returns safe default when reasoning fails."""
        return {
            "verdict"            : "UNVERIFIABLE",
            "confidence"         : 0.0,
            "reasoning"          : "Reasoning failed — marked unverifiable",
            "max_similarity"     : 0.0,
            "hallucination_flag" : False
        }

    def aggregate_sub_claim_verdicts(self, sub_claim_results):
        """
        Combines verdicts from all sub-claims into one final verdict.
        
        Aggregation logic:
        - If ANY sub-claim is FALSE → overall = FAKE
        - If ALL sub-claims are TRUE → overall = REAL
        - Otherwise → UNVERIFIABLE
        
        Why this logic?
        A compound claim is only fully true if ALL parts are true.
        One false sub-claim makes the whole claim misleading.
        This matches how fact-checkers reason.
        
        Args:
            sub_claim_results : list of reason() output dicts
            
        Returns:
            dict with final verdict, confidence, summary
        """

        verdicts    = [r["verdict"] for r in sub_claim_results]
        confidences = [r["confidence"] for r in sub_claim_results]
        hallucinated = any(r["hallucination_flag"] for r in sub_claim_results)

        # Aggregation logic
        if "FALSE" in verdicts:
            final_verdict    = "FALSE"
            # Confidence = average confidence of FALSE verdicts
            false_confs      = [
                r["confidence"] for r in sub_claim_results
                if r["verdict"] == "FALSE"
            ]
            final_confidence = sum(false_confs) / len(false_confs)

        elif all(v == "TRUE" for v in verdicts):
            final_verdict    = "TRUE"
            final_confidence = sum(confidences) / len(confidences)

        else:
            final_verdict    = "UNVERIFIABLE"
            final_confidence = sum(confidences) / len(confidences)

        return {
            "final_verdict"      : final_verdict,
            "final_confidence"   : round(final_confidence, 4),
            "hallucination_flag" : hallucinated,
            "sub_claim_count"    : len(sub_claim_results),
            "verdicts_breakdown" : verdicts
        }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from modules.retriever import Retriever

    retriever = Retriever()
    reasoner  = EvidenceReasoner()

    test_cases = [
        "COVID vaccines cause infertility",
        "COVID vaccines are banned in Europe",
        "The CDC offices in Atlanta will self-destruct in an emergency"
    ]

    print("="*60)
    print("  REASONING TEST")
    print("="*60)

    for claim in test_cases:
        print(f"\nSub-claim: {claim}")
        print("-"*50)

        evidence = retriever.retrieve(claim, top_k=3)
        result   = reasoner.reason(claim, evidence)

        print(f"  Verdict     : {result['verdict']}")
        print(f"  Confidence  : {result['confidence']}")
        print(f"  Similarity  : {result['max_similarity']}")
        print(f"  Hallucinated: {result['hallucination_flag']}")
        print(f"  Reasoning   : {result['reasoning']}")

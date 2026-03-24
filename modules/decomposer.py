# modules/decomposer.py
# Breaks compound claims into atomic verifiable sub-claims.
#
# This is Novel Contribution C1 of EvidenceChain.
#
# Why decompose?
# Compound claims contain multiple assertions.
# "X causes Y and is banned in Z" = two separate facts.
# Verifying them together produces mixed, unreliable evidence.
# Verifying each atomically gives precise, traceable verdicts.

import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.llm_client import LLMClient

load_dotenv()


class ClaimDecomposer:
    """
    Uses LLM to decompose compound claims into atomic sub-claims.
    
    Why LLM for decomposition?
    Rule-based decomposition (splitting on "and", "but") misses
    implicit compound claims.
    Example: "The president's policy destroyed the economy"
    → implicit claims: policy existed, economy was destroyed, 
      policy caused it
    LLM understands these implicit relationships.
    """

    def __init__(self):
        self.client = LLMClient()

    def decompose(self, claim):
        """
        Decomposes a claim into atomic verifiable sub-claims.
        
        Args:
            claim : raw claim string
            
        Returns:
            list of atomic sub-claim strings
        """

        # Why temperature=0.0?
        # Decomposition must be deterministic — same claim
        # must always produce same sub-claims for reproducibility
        prompt = f"""You are a fact-checking assistant. Your task is to break down a claim into atomic, independently verifiable statements.

Rules:
1. Each sub-claim must be a single verifiable fact
2. Do not add information not present in the original claim
3. Do not interpret or judge — only decompose
4. If the claim is already atomic, return it as is
5. Return ONLY a JSON array of strings, nothing else

Claim: "{claim}"

        Return format: ["sub-claim 1", "sub-claim 2", ...]"""

        try:
            raw = self.client.generate_json_text(prompt)

            # Parse JSON response
            # Why JSON? Structured output is parseable — no regex needed
            sub_claims = json.loads(raw)

            # Validate — must be a list of strings
            if not isinstance(sub_claims, list):
                return [claim]

            # Filter empty strings
            sub_claims = [s.strip() for s in sub_claims if s.strip()]

            return sub_claims if sub_claims else [claim]

        except json.JSONDecodeError:
            # If LLM returns non-JSON, fall back to original claim
            # Defensive programming — pipeline must never crash
            print(f"  Warning: JSON parse failed, using original claim")
            return [claim]

        except Exception as e:
            print(f"  Warning: Decomposition failed — {e}")
            return [claim]


if __name__ == "__main__":

    decomposer = ClaimDecomposer()

    # Test with compound claims from LIAR dataset
    test_claims = [
        "COVID vaccines cause infertility and are banned in Europe",
        "The unemployment rate is the lowest it has been in 50 years",
        "In the case of a catastrophic event, the Atlanta-area offices of the Centers for Disease Control and Prevention will self-destruct",
        "Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran",
        "The Chicago Bears have had more starting quarterbacks in the last 10 years than the total number of tenured faculty fired during the last two decades"
    ]

    print("="*60)
    print("  CLAIM DECOMPOSITION TEST")
    print("="*60)

    for claim in test_claims:
        print(f"\nOriginal claim:")
        print(f"  {claim}")
        print(f"\nSub-claims:")
        sub_claims = decomposer.decompose(claim)
        for i, sc in enumerate(sub_claims):
            print(f"  {i+1}. {sc}")
        print("-"*60)

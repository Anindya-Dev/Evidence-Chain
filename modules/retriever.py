# modules/retriever.py
# Retrieves evidence from FAISS knowledge base for a given claim.
# Applies source credibility + temporal weighting to results.

import os
import sys
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Retriever:
    """
    Loads FAISS index and retrieves weighted evidence for claims.
    Built once, used for every claim in the pipeline.
    """

    def __init__(self):
        print("Loading retriever...")

        # Load embedding model — same one used to build the index
        # Must be identical — different model = incompatible vectors
        self.embedder = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        kb_dir = config.get_knowledge_base_dir()

        # Load FAISS index from disk
        index_path    = os.path.join(kb_dir, "faiss_index.bin")
        self.index    = faiss.read_index(index_path)

        # Load metadata — source, date, weights for each document
        meta_path     = os.path.join(kb_dir, "metadata.pkl")
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"  Knowledge base : {kb_dir}")
        print(f"  Index loaded  : {self.index.ntotal} vectors")
        print(f"  Metadata      : {len(self.metadata)} documents")


    def retrieve(self, claim, top_k=None, use_weighting=True):
        """
        Retrieves top-k evidence chunks for a claim.
        Applies combined score = similarity x source_weight + recency_weight
        unless weighting is explicitly disabled for ablation.

        Args:
            claim  : string — the claim to retrieve evidence for
            top_k         : int  — number of results (default from config)
            use_weighting : bool — whether to apply source/recency weighting

        Returns:
            List of dicts with text, source, scores, combined_score
        """

        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        # Step 1 — Embed the claim
        # Convert claim text to 384-dim vector
        claim_embedding = self.embedder.encode([claim])
        claim_embedding = np.array(claim_embedding).astype("float32")

        # Normalize for cosine similarity — same as index normalization
        faiss.normalize_L2(claim_embedding)

        # Step 2 — Search FAISS
        # Returns similarity scores and indices of top matches
        # scores shape: (1, top_k), indices shape: (1, top_k)
        scores, indices = self.index.search(claim_embedding, top_k)

        # Step 3 — Build results with weighted scoring
        results = []
        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:   # FAISS returns -1 for empty slots
                continue

            meta = self.metadata[idx]

            # Combined score formula:
            # similarity x source_weight + recency_weight
            # Why multiply similarity by source_weight?
            # A highly similar but unreliable source should score lower
            # than a slightly less similar but authoritative source
            if use_weighting:
                combined_score = (
                    float(score) * meta["source_weight"] +
                    meta["recency_weight"]
                )
            else:
                combined_score = float(score)

            results.append({
                "text"           : meta["text"],
                "source"         : meta["source"],
                "date"           : meta["date"],
                "similarity"     : round(float(score), 4),
                "source_weight"  : meta["source_weight"],
                "recency_weight" : meta["recency_weight"],
                "combined_score" : round(combined_score, 4)
            })

        # Sort by combined score — best evidence first
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        return results


    def retrieve_for_subclaims(self, subclaims):
        """
        Retrieves evidence for each sub-claim independently.
        This is the core of our claim decomposition approach.

        Args:
            subclaims : list of strings

        Returns:
            Dict mapping each subclaim to its evidence list
        """

        evidence_map = {}
        for subclaim in subclaims:
            evidence_map[subclaim] = self.retrieve(subclaim)
        return evidence_map


if __name__ == "__main__":
    # Test retriever with sample claims from LIAR

    retriever = Retriever()

    test_claims = [
        "COVID vaccines cause infertility and are banned in Europe",
        "The unemployment rate is at a 50 year low",
        "The CDC headquarters will self destruct in an emergency"
    ]

    for claim in test_claims:
        print(f"\n{'='*60}")
        print(f"Claim: {claim}")
        print(f"{'='*60}")

        results = retriever.retrieve(claim, top_k=3)

        for i, r in enumerate(results):
            print(f"\nEvidence {i+1}:")
            print(f"  Source         : {r['source']}")
            print(f"  Date           : {r['date']}")
            print(f"  Similarity     : {r['similarity']}")
            print(f"  Source weight  : {r['source_weight']}")
            print(f"  Combined score : {r['combined_score']}")
            print(f"  Text           : {r['text'][:120]}...")

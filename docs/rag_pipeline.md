## EvidenceChain RAG Pipeline

### Offline step: build the knowledge base

1. Collect trusted evidence documents.
2. Encode each document with the sentence embedding model.
3. Store embeddings in a FAISS index.
4. Save metadata such as source, date, source weight, and recency weight.

Current implementation note:

- The saved repository index is document-level.
- Chunk size and overlap are already defined in config but are not yet
  used in the saved knowledge base.

### Online step: verify a claim

1. Preprocess the claim.
2. Decompose the claim into sub-claims.
3. Convert each sub-claim into an embedding.
4. Search FAISS for the top-k most similar evidence items.
5. Rank evidence with similarity plus source and recency weighting.
6. Send the sub-claim and evidence to the LLM.
7. Receive verdict, confidence, and explanation.
8. Check for hallucination using similarity and confidence.
9. Aggregate sub-claim verdicts into a final RAG verdict.

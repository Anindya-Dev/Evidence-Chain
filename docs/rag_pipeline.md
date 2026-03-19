BEFORE any claim arrives:
  1. Load trusted documents (Wikipedia, news articles)
  2. Split into chunks (300 words each)
  3. Convert chunks to embeddings (sentence-transformers)
  4. Store embeddings in FAISS index
  → This is the knowledge base — built ONCE, used forever

WHEN a claim arrives:
  1. Convert claim to embedding
  2. Search FAISS for top-5 similar chunks
  3. Apply source credibility weight to each chunk
  4. Apply temporal weight (prefer recent)
  5. Feed claim + evidence to LLM
  6. LLM returns verdict + reasoning
  7. Check for hallucination
  8. Return structured result
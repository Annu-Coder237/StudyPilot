import math
import re
from collections import Counter, defaultdict


def _tokens(text):
    # Keep common equation-like tokens such as x[n], x[k], x[n-k], H(z).
    return re.findall(r"[a-zA-Z]+(?:\[[^\]\s]+\])?|[0-9]+(?:\.[0-9]+)?|[∑∫√∞≈≤≥∂∇∏±×÷]|[a-zA-Z]+", (text or "").lower())


def retrieve_relevant_chunks(query, chunks, top_k=5):
    """
    Lightweight local TF-IDF retrieval. No API key, embedding model, or network required.
    Returns chunk dictionaries with a retrieval_score field.
    """
    if not query or not query.strip() or not chunks:
        return []

    docs = [str(chunk.get("text", "")) for chunk in chunks]
    token_docs = [_tokens(doc) for doc in docs]
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    n_docs = len(docs)
    document_frequency = Counter()
    term_counts = []
    for tokens in token_docs:
        counts = Counter(tokens)
        term_counts.append(counts)
        document_frequency.update(set(tokens))

    def idf(token):
        return math.log(1 + (n_docs + 1) / (document_frequency.get(token, 0) + 1))

    query_counts = Counter(query_tokens)
    query_vec = {t: (1 + math.log(c)) * idf(t) for t, c in query_counts.items()}
    query_norm = math.sqrt(sum(v * v for v in query_vec.values())) or 1.0

    scored = []
    for original, counts in zip(chunks, term_counts):
        doc_vec = {t: (1 + math.log(c)) * idf(t) for t, c in counts.items()}
        doc_norm = math.sqrt(sum(v * v for v in doc_vec.values())) or 1.0
        dot = sum(weight * doc_vec.get(token, 0.0) for token, weight in query_vec.items())
        score = dot / (query_norm * doc_norm)
        # Small exact phrase bonus helps equation terms and named concepts.
        normalized_query = " ".join(query.lower().split())
        normalized_doc = " ".join(str(original.get("text", "")).lower().split())
        if len(normalized_query) > 3 and normalized_query in normalized_doc:
            score += 0.15
        if score > 0:
            item = dict(original)
            item["retrieval_score"] = float(score)
            scored.append(item)

    scored.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return scored[:max(1, int(top_k))]

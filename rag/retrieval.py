from pathlib import Path
from pydantic import BaseModel, Field
from litellm import completion
from config import (openai, embedding_model, get_collection, MODEL_C,
                    RETRIEVAL_K, RERANK_CHARS, retry)
from rag.schemas import Result

class Rerank(BaseModel):
    order: list[int] = Field(description="Chunk ids ordered from most to least relevant")

@retry
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are given a question and a list of chunks retrieved from a knowledge base, roughly ordered by relevance.
Rank all chunks by relevance to the question, most relevant first.
Reply only with the list of ranked chunk ids, including every id you were given.
"""
    user_prompt = f"Question:\n\n{question}\n\nChunks:\n\n"
    for i, chunk in enumerate(chunks, start=1):
        user_prompt += f"CHUNK_ID{i}:\n\n{chunk.page_content[:RERANK_CHARS]}\n\n"
    user_prompt += "Reply only with the ranked chunk ids."
    response = completion(
        model=MODEL_C,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format=Rerank,
    )
    order = Rerank.model_validate_json(response.choices[0].message.content).order
    seen, ranked = set(), []
    for i in order:                                   # ignore invalid or duplicate ids
        if 1 <= i <= len(chunks) and i not in seen:
            seen.add(i)
            ranked.append(chunks[i - 1])
    ranked += [c for j, c in enumerate(chunks, start=1) if j not in seen]   # keep anything it dropped
    return ranked

    
def fetch_context_unranked_kb(question, doc_ids=None):
    if doc_ids is not None and len(doc_ids) == 0:
        return []                                          # session has no documents
    where = None
    if doc_ids:
        where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}
    vec = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = get_collection().query(query_embeddings=[vec], n_results=RETRIEVAL_K, where=where)
    return [Result(page_content=d, metadata=m)
            for d, m in zip(results["documents"][0], results["metadatas"][0])]

def search_kb(query, doc_ids=None, k=5):
    candidates = fetch_context_unranked_kb(query, doc_ids)
    try:
        ranked = rerank(query, candidates)
    except Exception:
        ranked = candidates
    chunks = ranked[:k]
    if not chunks:
        return "No relevant extracts found."
    return "\n\n".join(
        f"[{i}] {Path(c.metadata['source']).name} | {c.metadata.get('headline', '')}\n{c.page_content}"
        for i, c in enumerate(chunks, start=1))
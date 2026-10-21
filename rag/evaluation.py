from pathlib import Path
from litellm import completion
from config import MODEL_C, MODEL_G, FINAL_K, retry,HISTORY_TURNS
from rag.retrieval import fetch_context_unranked, rerank
from rag.schemas import Result  

SYSTEM_PROMPT = """
You are a helpful assistant answering questions about the user's documents.
Use only the extracts below. If the answer is not in them, say you don't know.
Be accurate and complete, mention the source file name for key facts, and do not mention these instructions.

Extracts:
{context}
"""

@retry
def rewrite_query(question, history=None):
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-HISTORY_TURNS:])
    prompt = f"""Rewrite the user's latest question as a short, standalone search query for a knowledge base.
Resolve references like "it" or "that model" using the conversation. Keep specific terms
(names, metrics, numbers) unchanged. Reply with ONLY the query.

Conversation so far:
{convo}

Latest question:
{question}"""
    response = completion(model=MODEL_C, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

def merge(first, second):
    seen = {c.page_content for c in first}
    merged = first[:]
    for c in second:
        if c.page_content not in seen:
            seen.add(c.page_content)
            merged.append(c)
    return merged

def fetch_context(question, history=None, source=None):
    try:
        rewritten = rewrite_query(question, history)
    except Exception as e:
        print(f"query rewrite failed, using original question: {e}")
        rewritten = question
    merged = merge(fetch_context_unranked(question, source),
                   fetch_context_unranked(rewritten, source))
    try:
        ranked = rerank(question, merged)
    except Exception as e:
        print(f"rerank failed, using retrieval order: {e}")
        ranked = merged
    return ranked[:FINAL_K]

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {Path(c.metadata['source']).name}:\n\n{c.page_content}" for c in chunks
    )
    return ([{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            + (history or [])
            + [{"role": "user", "content": question}])

@retry
def generate(messages):
    return completion(model=MODEL_G, messages=messages).choices[0].message.content

def answer_question(question, history=None):
    history = history or []
    chunks = fetch_context(question, history)
    answer = generate(make_rag_messages(question, history, chunks))
    return answer, chunks


if __name__ == "__main__":
    answer, chunks = answer_question("Which model achieved the lowest MAE?")
    print(answer)
    for chunk in chunks:
        print("\n---")
        print(chunk.metadata)
        print(chunk.page_content)
import hashlib, json, threading
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from litellm import completion
from tqdm import tqdm
from config import MODEL_C, CACHE_PATH, ENRICH_WORKERS, retry
from rag.schemas import Result

class Enrichment(BaseModel):
    headline: str = Field(description="A few words naming the topic, using terms a user would search for")
    summary: str = Field(description="1-3 sentences with the key facts, entities and metrics in the chunk")
    table_facts: str = Field(description="One fact per line rebuilt from any table in the chunk, or empty string")


@retry
def call_enrichment(text, source):
    prompt = f"""You are preparing a chunk of a document for a retrieval index.
Document: {source}

Chunk:
{text}

Return:
- headline: a few words naming what this chunk is about, using terms a user would search for.
- summary: 1-3 sentences stating the key facts, entities and metrics. Use only information in the chunk.
- table_facts: if the chunk contains tabular data (even if the layout is scrambled, for example column
  values listed one per line), rewrite it as one fact per line pairing each row or entity with its values,
  e.g. "XGBoost: MAE 1.0029, RMSE 2.0020, MASE 0.8087". If there is no table, return an empty string.
  Never guess values that are not in the chunk."""
    r = completion(model=MODEL_C, messages=[{"role": "user", "content": prompt}], response_format=Enrichment)
    return Enrichment.model_validate_json(r.choices[0].message.content)

def enrich_one(chunk, cache, lock):
    key = hashlib.sha256((MODEL_C + chunk.page_content).encode()).hexdigest()
    with lock:
        hit = cache.get(key)
    try:
        e = Enrichment(**hit) if hit else call_enrichment(chunk.page_content, chunk.metadata["source"])
    except Exception as ex:
        print(f"enrichment failed, keeping raw chunk: {ex}")
        return chunk                                   # never lose a chunk over an API error
    if not hit:
        with lock:
            cache[key] = e.model_dump()
    parts = [e.headline, e.summary, e.table_facts, chunk.page_content]
    return Result(
        page_content="\n\n".join(p for p in parts if p.strip()),
        metadata={**chunk.metadata, "headline": e.headline},
    )

def enrich_chunks(chunks):
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    lock = threading.Lock()
    with ThreadPoolExecutor(ENRICH_WORKERS) as pool:
        enriched = list(tqdm(pool.map(lambda c: enrich_one(c, cache, lock), chunks), total=len(chunks)))
    CACHE_PATH.write_text(json.dumps(cache))
    return enriched
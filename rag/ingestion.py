import hashlib
from pathlib import Path
from chromadb import PersistentClient
from multiprocessing import Pool
from tqdm import tqdm
from config import openai, embedding_model, KNOWLEDGE_BASE_PATH, WORKERS,collection_name, DB_NAME
from rag.loader import load_document
from rag.chunking import process_document
from rag.enrichment import enrich_chunks



def ingest_file(path: Path) -> str:
    path = Path(path)
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    col = PersistentClient(path=DB_NAME).get_or_create_collection(collection_name)
    if col.get(where={"doc_id": doc_id}, limit=1)["ids"]:
        return doc_id                                     # same file already indexed
    document = {"type": path.suffix.lower().lstrip("."), "source": path.name,
                "text": load_document(path)}
    chunks = process_document(document)                   # no Pool for a single file
    if not chunks:
        raise ValueError("No text could be extracted (scanned or image-only file?)")
    chunks = enrich_chunks(chunks)
    for c in chunks:
        c.metadata["doc_id"] = doc_id
    for start in range(0, len(chunks), 100):              # batch the embedding calls
        batch = chunks[start:start + 100]
        vecs = [e.embedding for e in openai.embeddings.create(
            model=embedding_model, input=[c.page_content for c in batch]).data]
        col.add(ids=[f"{doc_id}-{start + i}" for i in range(len(batch))],
                embeddings=vecs,
                documents=[c.page_content for c in batch],
                metadatas=[c.metadata for c in batch])
    return doc_id


def fetch_documents():
    """Fetches Documents from the Directory"""
    documents= []
    for file in KNOWLEDGE_BASE_PATH.rglob("*"):
        if file.is_file():
            markdown_text=load_document(file)
            documents.append({"type":file.suffix.lower().replace(".",""),"source":file.as_posix(), "text": markdown_text})
    print(f"loaded {len(documents)} documents") 
    return documents

def make_chunks(documents):
    """
    Create chunkers with workers who creats chunks in parallel

    """
    chunks=[]
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document,documents),total=len(documents)):
            chunks.extend(result)

    return chunks
def create_embeddings(chunks):
    chroma=PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts=[chunk.page_content for chunk in chunks]
    emb=openai.embeddings.create(model=embedding_model,input=texts).data
    vectors=[e.embedding for e in emb]

    collection=chroma.get_or_create_collection(collection_name)

    ids=[ str(i) for i in range(len(chunks))]
    metas= [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

if __name__== "__main__":
    documents=fetch_documents()
    chunks=make_chunks(documents)
    chunks = enrich_chunks(chunks)
    create_embeddings(chunks)
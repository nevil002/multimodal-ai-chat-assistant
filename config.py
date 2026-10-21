from dotenv import load_dotenv
from chromadb import PersistentClient
from tenacity import retry, wait_exponential, stop_after_attempt
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import json
from pathlib import Path
from mcp import StdioServerParameters


load_dotenv(override=True)
openai_api_key=os.getenv('OPENAI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')


model_gpt='gpt-4o-mini'
model_groq="openai/gpt-oss-120b"
embedding_model = "text-embedding-3-large"
collection_name = "docs"
DB_NAME = str(Path(__file__).parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "Knowledge_Base"
CACHE_PATH = Path(__file__).parent / "enrich_cache.json"
MAX_CHUNK_SIZE = 4000
OVERLAP = 500
RETRIEVAL_K = 20
FINAL_K = 10
RERANK_CHARS = 1500
ENRICH_WORKERS = 8
WORKERS=3
MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 6
ROOT = Path(__file__).resolve().parent

MODEL_G = "openai/gpt-oss-120b"
MODEL_C = "gpt-4.1-nano"

MCP_SERVERS = {
    "utilities": StdioServerParameters(
        command=sys.executable,                       # same venv Python as the app
        args=[str(ROOT / "mcp_servers" / "utility_server.py")]),
}

openai=OpenAI()
groq=OpenAI(api_key=groq_api_key,base_url="https://api.groq.com/openai/v1")
chroma = PersistentClient(path=DB_NAME)
retry = retry(wait=wait_exponential(multiplier=1, min=2, max=30),
                  stop=stop_after_attempt(3), reraise=True)

def get_collection():
    # fetched per call so a re-ingestion (delete + recreate) never leaves a stale handle
    return chroma.get_collection(collection_name)



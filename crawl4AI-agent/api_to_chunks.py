from dataclasses import dataclass
import os
import json
import numpy as np
import faiss
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client
import json
from datetime import datetime, timezone
import asyncio
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

load_dotenv()
def get_required_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value
# Type-safe access to env vars
SUPABASE_URL = get_required_env("SUPABASE_URL")
SUPABASE_SERVICE_KEY = get_required_env("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = get_required_env("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# async def get_embedding(text: str) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
#         response = await openai_client.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return [0] * 1536  # Return zero vector on error


# @dataclass
# class ProcessedChunk:
#     identifier: str
#     chunk_number: int
#     title: str
#     summary: str
#     content: str
#     metadata: Dict[str, Any]
#     embedding: List[float]
    
# async def insert_chunk(chunk: ProcessedChunk):
#     """Insert a processed chunk into Supabase."""
#     try:
#         data = {
#             "identifier": chunk.identifier,
#             "chunk_number": chunk.chunk_number,
#             "title": chunk.title,
#             "summary": chunk.summary,
#             "content": chunk.content,
#             "metadata": chunk.metadata,
#             "embedding": chunk.embedding
#         }
        
#         result = supabase.table("site_pages").insert(data).execute()
#         print(f"Inserted chunk {chunk.chunk_number} for {chunk.identifier}")
#         return result
#     except Exception as e:
#         print(f"Error inserting chunk: {e}")
#         return None


# async def prepare_processed_chunks(api_chunks):
#     processed_chunks = []

#     for idx, chunk in enumerate(api_chunks):
#         # Create identifier
#         identifier = "-".join(chunk['path'])

#         # Stringify the content
#         content = (
#             f"API Path: {' -> '.join(chunk['path'])}\n\n"
#             f"Request:\n{json.dumps(chunk['request'], indent=2)}\n\n"
#             f"Response:\n{json.dumps(chunk['response'], indent=2)}"
#         )

#         # Metadata
#         metadata = {
#             "source": "sikka_api_docs",
#             "chunk_size": len(content),
#             "crawled_at": datetime.now(timezone.utc).isoformat(),
#             "identifier": identifier
#         }

#         # Placeholder for title/summary - you will use LLM here
#         title = "PLACEHOLDER_TITLE"
#         summary = "PLACEHOLDER_SUMMARY"

#         # Get embedding
#         embedding = await get_embedding(chunk)

#         # Assemble
#         processed_chunk = {
#             "identifier": identifier,
#             "chunk_number": idx,
#             "title": title,
#             "summary": summary,
#             "content": content,
#             "metadata": metadata,
#             "embedding": None  # You will fill this separately
#         }

#         processed_chunks.append(processed_chunk)

#     return processed_chunks

# def find_api_items(obj, path=None, collected=None):
#     if collected is None:
#         collected = []
#     if path is None:
#         path = []

#     if isinstance(obj, dict):
#         if 'request' in obj and 'response' in obj:
#             collected.append({
#                 "path": path.copy(),
#                 "request": obj['request'],
#                 "response": obj['response'],
#             })
#         else:
#             for k, v in obj.items():
#                 find_api_items(v, path + [k], collected)
#     elif isinstance(obj, list):
#         for idx, item in enumerate(obj):
#             find_api_items(item, path + [str(idx)], collected)

#     return collected

# async def load_one_api():
#     with open("sikka-apis.json", "r", encoding="utf-8") as f:
#         postman_data = json.load(f)
#         api_chunks = find_api_items(postman_data)
#         processed_chunks = await prepare_processed_chunks(api_chunks)
#         for chunk in processed_chunks:
#           print("────────────────────────────────────────")
#           print(f"🔹 Chunk #{chunk['chunk_number']}")
#           print(f"🔸 Title   : {chunk['title']}")
#           print(f"🔸 Summary : {chunk['summary']}")
#           print(f"🔸 URL     : {chunk['identifier']}")
#           print(f"🔸 Length  : {len(chunk['content'])} chars")
#           print(f"🔸 Metadata: {chunk['metadata']}")
#           embedding_info = f"Vector of length {len(chunk['embedding'])}" if chunk['embedding'] else "Not generated yet"
#           print(f"🔸 Embedding: {embedding_info}")
#           print("🔸 Content Preview:\n", chunk['content'][:300], "...\n")
#           print("────────────────────────────────────────\n")

        # return postman_data['item'][0]['item'][2]['item']


async def collect_docs(node):
    """
    Depth‑first walk over *any* Postman collection object (dict OR list) and
    return a list of {content:str, path:str}.
    """
    stack = [([], node)]
    out   = []

    while stack:
        path, cur = stack.pop()

        # ── handle dict nodes ───────────────────────────────────────────────
        if isinstance(cur, dict):
            name     = cur.get("name") or "<no‑name>"
            new_path = path + [name]

            buckets = []

            # 1) plain description
            if cur.get("description"):
                buckets.append(str(cur["description"]))

            # 2) request‑level docs + sample body
            req = cur.get("request", {})
            if isinstance(req, dict):
                if req.get("description"):
                    buckets.append(req["description"])
                raw = req.get("body", {}).get("raw")
                if raw: buckets.append("```json\n"+raw[:2000]+"\n```")

            # 3) response sample bodies
            for resp in cur.get("response", []):
                body = resp.get("body")
                if body:
                    buckets.append("```json\n"+body[:2000]+"\n```")

            # 4) event → script → exec (often holds markdown examples)
            for ev in cur.get("event", []):
                exec_lines = ev.get("script", {}).get("exec") or []
                if exec_lines:
                    buckets.append("\n".join(exec_lines)[:2000])

            # save doc
            if buckets:
                out.append({
                    "content": f"# {' › '.join(new_path)}\n\n" + "\n\n".join(buckets),
                    "path":    " › ".join(new_path)
                })

            # push children (if any)
            for child in cur.get("item", []):
                stack.append((new_path, child))

        # ── handle list nodes ───────────────────────────────────────────────
        elif isinstance(cur, list):
            for itm in cur:
                stack.append((path, itm))

    return out

async def embed_texts(texts: list[str], model: str = "text-embedding-3-small"):
    """
    Returns a list[ list[float] ] of embeddings using the new OpenAI ≥1.0.0 SDK.
    """
    client = OpenAI(api_key=OPENAI_API_KEY) 
    resp = client.embeddings.create(model=model, input=texts)
    # resp.data is a list of Embedding objects in **input order**
    return [e.embedding for e in resp.data]

async def main():
    with open("sikka-apis.json", "r", encoding="utf-8") as f:
        postman_data = json.load(f)
        docs = await collect_docs(postman_data)
        print("Total docs captured:", len(docs))
        print("\nFirst snippet ↓\n", docs[0]["content"][:300])

        vecs, BATCH = [], 96
        for i in range(0, len(docs), BATCH):
            batch = [d["content"] for d in docs[i:i+BATCH]]
            vecs.extend(await embed_texts(batch))  # No await

        vecs = np.asarray(vecs, dtype="float32")
        print(f"vecs shape: {vecs.shape}")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)  # type: ignore
        print("FAISS index size:", index.ntotal)


if __name__ == "__main__":
    asyncio.run(main())
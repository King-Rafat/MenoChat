from FlagEmbedding import BGEM3FlagModel, FlagReranker
import numpy as np
import json
from pathlib import Path
import faiss
import soundfile as sf

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

def load_vector_db(faiss_index_path, chunks_path, meta_path, uid_list_path):
    print("Loading vector database...")
    index = faiss.read_index(faiss_index_path)
    chunks = load_jsonl(chunks_path)
    meta = load_jsonl(meta_path)

    print(f"FAISS ntotal: {index.ntotal}, Chunks: {len(chunks)}, Meta: {len(meta)}")
    assert index.ntotal == len(chunks) == len(meta), "DB mapping mismatch!"
    return index, chunks, meta

def load_embedding_models(emb_model_dir, rerank_model_dir):
    print("Loading embedding and reranking models...")
    embedder = BGEM3FlagModel(emb_model_dir, use_fp16=False, devices='cpu')
    reranker = FlagReranker(rerank_model_dir, use_fp16=False, devices='cuda')
    print("Embedding and reranking models loaded successfully!")
    return embedder, reranker

def l2_normalize(v):
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom

def embed_query(embedder, q):
    out = embedder.encode([q], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    vec = np.array(out["dense_vecs"], dtype=np.float32)
    return l2_normalize(vec).astype("float32")

def retrieve_then_rerank(embedder, reranker, index, chunks, meta, question_bn, top_k=3, faiss_top_n=5):
    """RAG: Retrieve from FAISS and rerank"""
    q_for_retrieval = question_bn
    
    qvec = embed_query(embedder, q_for_retrieval)
    D, I = index.search(qvec, faiss_top_n)
    cand_idxs = I[0].tolist()
    cand_texts = []
    valid_idxs = []
    
    for i in cand_idxs:
        t = chunks[i].get("text") or chunks[i].get("chunk_text") or ""
        t = (t or "").strip()
        if t:
            cand_texts.append(t)
            valid_idxs.append(i)

    if not valid_idxs:
        return [], ""

    pairs = [[q_for_retrieval, cand_texts[j]] for j in range(len(valid_idxs))]
    scores = np.array(reranker.compute_score(pairs), dtype=np.float32)
    order = np.argsort(-scores)[:top_k]
    
    top = []
    for j in order:
        idx = valid_idxs[j]
        md = meta[idx].get("metadata", {}) if isinstance(meta[idx], dict) else {}
        top.append({
            "idx": idx,
            "text": cand_texts[j],
            "rerank_score": float(scores[j]),
            "metadata": md,
        })
    
    ctx = "\n\n---\n\n".join([t["text"] for t in top])
    return top, ctx

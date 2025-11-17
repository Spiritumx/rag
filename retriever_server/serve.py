from time import perf_counter
from fastapi import FastAPI, Request
import os
from pathlib import Path
from typing import Tuple, Optional, Dict

from unified_retriever import UnifiedRetriever


def _load_dense_components(model_source: Optional[str]) -> Tuple[Optional[object], Optional[object]]:
    if not model_source:
        return None, None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_source)
        return model, None
    except Exception:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModel.from_pretrained(model_source)
        return model, tokenizer


def _load_splade_components(model_source: Optional[str]) -> Tuple[Optional[object], Optional[object]]:
    if not model_source:
        return None, None
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForMaskedLM.from_pretrained(model_source)
    return model, tokenizer


def _load_reranker(model_source: Optional[str], device: Optional[str]) -> Optional[object]:
    if not model_source:
        return None
    try:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(model_source, device=device)
        return reranker
    except Exception as exc:
        print(f"[Retriever] Failed to load reranker {model_source}: {exc}")
        return None


def _default_model_path(*parts) -> Optional[str]:
    candidate = Path(__file__).resolve().parent.joinpath("models", *parts)
    if candidate.exists():
        return str(candidate)
    return None


def _resolve_model_source(env_path_key: str, env_name_key: str, default_subdir: Tuple[str, ...], default_name: str) -> Optional[str]:
    if os.environ.get(env_path_key):
        return os.environ[env_path_key]
    if os.environ.get(env_name_key):
        return os.environ[env_name_key]
    local_path = _default_model_path(*default_subdir, default_name)
    if local_path:
        return local_path
    return None


def _parse_hybrid_weights(value: Optional[str]) -> Optional[Dict[str, float]]:
    if not value:
        return None
    weights = {}
    for segment in value.split(","):
        if not segment.strip():
            continue
        if ":" not in segment:
            continue
        key, val = segment.split(":", 1)
        try:
            weights[key.strip()] = float(val.strip())
        except ValueError:
            continue
    return weights or None


dense_source = _resolve_model_source(
    env_path_key="RETRIEVER_DENSE_MODEL_PATH",
    env_name_key="RETRIEVER_DENSE_MODEL_NAME",
    default_subdir=("dense",),
    default_name="all-MiniLM-L6-v2",
)
splade_source = _resolve_model_source(
    env_path_key="RETRIEVER_SPLADE_MODEL_PATH",
    env_name_key="RETRIEVER_SPLADE_MODEL_NAME",
    default_subdir=("splade",),
    default_name="splade-v3",
)
reranker_source = _resolve_model_source(
    env_path_key="RETRIEVER_RERANKER_MODEL_PATH",
    env_name_key="RETRIEVER_RERANKER_MODEL_NAME",
    default_subdir=("reranker",),
    default_name="cross-encoder-ms-marco-MiniLM-L-6-v2",
)

dense_model = None
dense_tokenizer = None
splade_model = None
splade_tokenizer = None
reranker_model = None
device_override = os.environ.get("RETRIEVER_DEVICE")

if dense_source:
    dense_model, dense_tokenizer = _load_dense_components(dense_source)
    print(f"[Retriever] Loaded dense model from {dense_source}")
else:
    print("[Retriever] Dense model not provided. HNSW retrieval disabled.")

if splade_source:
    splade_model, splade_tokenizer = _load_splade_components(splade_source)
    print(f"[Retriever] Loaded SPLADE model from {splade_source}")
else:
    print("[Retriever] SPLADE model not provided. SPLADE retrieval disabled.")

if reranker_source:
    reranker_model = _load_reranker(reranker_source, device=device_override)
    if reranker_model is not None:
        print(f"[Retriever] Loaded reranker model from {reranker_source}")
    else:
        print(f"[Retriever] Failed to load reranker from {reranker_source}")
else:
    print("[Retriever] Reranker model not provided. Hybrid ranking will rely on ES scores only.")

hybrid_weights = _parse_hybrid_weights(os.environ.get("RETRIEVER_HYBRID_WEIGHTS"))
rerank_top_k = int(os.environ.get("RETRIEVER_RERANK_TOP_K", "50"))

retriever = UnifiedRetriever(
    host=os.environ.get("RETRIEVER_ES_HOST", "http://localhost/"),
    port=int(os.environ.get("RETRIEVER_ES_PORT", 9200)),
    dense_model=dense_model,
    dense_tokenizer=dense_tokenizer,
    splade_model=splade_model,
    splade_tokenizer=splade_tokenizer,
    device=device_override,
    hybrid_weights=hybrid_weights,
    reranker=reranker_model,
    rerank_top_k=rerank_top_k,
)

app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}


@app.post("/retrieve/")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch",)
    start_time = perf_counter()
    retrieval = getattr(retriever, retrieval_method)(**arguments)
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}

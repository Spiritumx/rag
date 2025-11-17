from typing import List, Dict, Optional
import argparse

from collections import OrderedDict
from elasticsearch import Elasticsearch

import numpy as np
import torch


class ElasticsearchRetriever:
    """
    Some useful resources for constructing ES queries:
    # https://stackoverflow.com/questions/28768277/elasticsearch-difference-between-must-and-should-bool-query
    # https://stackoverflow.com/questions/49826587/elasticsearch-query-to-match-two-different-fields-with-exact-values

    # bool/must acts as AND
    # bool/should acts as OR
    # bool/filter acts as binary filter w/o score (unlike must and should).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        dense_model=None,
        dense_tokenizer=None,
        splade_model=None,
        splade_tokenizer=None,
        device: Optional[str] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        reranker=None,
        rerank_top_k: int = 50,
    ):
        self._es = Elasticsearch([host], scheme="http", port=port, timeout=30)
        self.dense_model = dense_model
        self.dense_tokenizer = dense_tokenizer
        self.splade_model = splade_model
        self.splade_tokenizer = splade_tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hybrid_weights = hybrid_weights or {"bm25": 1.0, "dense": 1.0, "splade": 1.0}
        self.reranker = reranker
        self.rerank_top_k = max(1, rerank_top_k)

        if isinstance(self.dense_model, torch.nn.Module):
            self.dense_model.to(self.device)
            self.dense_model.eval()
        if self.splade_model is not None:
            self.splade_model.to(self.device)
            self.splade_model.eval()

    def retrieve_paragraphs(
        self,
        corpus_name: str,
        query_text: str = None,
        is_abstract: bool = None,
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        query_title_field_too: bool = False,
        paragraph_index: int = None,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
        retrieval_backend: str = "bm25",
        hybrid_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:

        retrieval_backend = (retrieval_backend or "bm25").lower()

        if retrieval_backend not in ("bm25", "hnsw", "splade", "hybrid"):
            raise ValueError(f"Unsupported retrieval_backend: {retrieval_backend}")

        if retrieval_backend == "bm25":
            hits = self._retrieve_bm25(
                corpus_name=corpus_name,
                query_text=query_text,
                is_abstract=is_abstract,
                allowed_titles=allowed_titles,
                allowed_paragraph_types=allowed_paragraph_types,
                query_title_field_too=query_title_field_too,
                paragraph_index=paragraph_index,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_hits_count,
                apply_reranker=True,
            )
        elif retrieval_backend == "hnsw":
            hits = self._retrieve_hnsw(
                corpus_name=corpus_name,
                query_text=query_text,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_hits_count,
                apply_reranker=True,
            )
        elif retrieval_backend == "splade":
            hits = self._retrieve_splade(
                corpus_name=corpus_name,
                query_text=query_text,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_hits_count,
                apply_reranker=True,
            )
        else:
            weights = self.hybrid_weights.copy()
            if hybrid_weights:
                weights.update(hybrid_weights)
            hits = self._retrieve_hybrid(
                corpus_name=corpus_name,
                query_text=query_text,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_hits_count,
                weights=weights,
            )

        if allowed_titles is not None:
            lower_allowed_titles = [e.lower().strip() for e in allowed_titles]
            hits = [item for item in hits if item["title"].lower().strip() in lower_allowed_titles]

        if allowed_paragraph_types is not None:
            hits = [
                item
                for item in hits
                if item.get("paragraph_type") in allowed_paragraph_types or not item.get("paragraph_type")
            ]

        for retrieval_ in hits:
            retrieval_["corpus_name"] = corpus_name
        return hits

    def _retrieve_bm25(
        self,
        corpus_name: str,
        query_text: str,
        is_abstract: bool,
        allowed_titles: List[str],
        allowed_paragraph_types: List[str],
        query_title_field_too: bool,
        paragraph_index: Optional[int],
        max_buffer_count: int,
        max_hits_count: int,
        apply_reranker: bool = True,
    ) -> List[Dict]:
        query = {
            "size": max_buffer_count,
            # what records are needed in result
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "should": [],
                    "must": [],
                }
            },
        }

        if query_text is not None:
            # must is too strict for this:
            query["query"]["bool"]["should"].append({"match": {"paragraph_text": query_text}})

        if query_title_field_too:
            query["query"]["bool"]["should"].append({"match": {"title": query_text}})

        if is_abstract is not None:
            query["query"]["bool"]["filter"] = [{"match": {"is_abstract": is_abstract}}]

        if allowed_titles is not None:
            if len(allowed_titles) == 1:
                query["query"]["bool"]["must"] += [{"match": {"title": _title}} for _title in allowed_titles]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _title}}}} for _title in allowed_titles
                ]

        if allowed_paragraph_types is not None:
            if len(allowed_paragraph_types) == 1:
                query["query"]["bool"]["must"] += [
                    {"match": {"paragraph_type": _paragraph_type}} for _paragraph_type in allowed_paragraph_types
                ]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _paragraph_type}}}}
                    for _paragraph_type in allowed_paragraph_types
                ]

        if paragraph_index is not None:
            query["query"]["bool"]["should"].append({"match": {"paragraph_index": paragraph_index}})

        assert query["query"]["bool"]["should"] or query["query"]["bool"]["must"]

        if not query["query"]["bool"]["must"]:
            query["query"]["bool"].pop("must")

        if not query["query"]["bool"]["should"]:
            query["query"]["bool"].pop("should")

        result = self._es.search(index=corpus_name, body=query)

        retrieval = self._dedupe_and_format_hits(result, max_hits_count)
        if apply_reranker:
            retrieval = self._apply_reranker(query_text, retrieval)
        return retrieval

    def retrieve_titles(
        self,
        query_text: str,
        corpus_name: str,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            # what records are needed in the result.
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "must": [
                        {"match": {"title": query_text}},
                    ],
                    "filter": [
                        {"match": {"is_abstract": True}},  # so that same title doesn't show up many times.
                    ],
                }
            },
        }

        result = self._es.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["title"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())[:max_hits_count]

        retrieval = [e["_source"] for e in retrieval]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval


    # --- private helpers ---

    def _dedupe_and_format_hits(self, result, max_hits_count: int) -> List[Dict]:
        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["paragraph_text"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())

        retrieval = sorted(retrieval, key=lambda e: e["_score"], reverse=True)
        retrieval = retrieval[:max_hits_count]
        formatted = []
        for retrieval_ in retrieval:
            source = retrieval_["_source"]
            source["score"] = retrieval_["_score"]
            source["es_doc_id"] = retrieval_["_id"]
            formatted.append(source)
        return formatted

    def _apply_reranker(self, query_text: Optional[str], hits: List[Dict]) -> List[Dict]:
        if self.reranker is None or not query_text or not hits:
            return hits
        rerank_size = min(self.rerank_top_k, len(hits))
        pairs = [(query_text, hit.get("paragraph_text", "")) for hit in hits[:rerank_size]]
        try:
            scores = self.reranker.predict(pairs)
        except Exception as exc:
            print(f"[Retriever] Reranker prediction failed: {exc}")
            return hits

        if isinstance(scores, list):
            scored = scores
        else:
            scored = scores.tolist()

        for hit, score in zip(hits[:rerank_size], scored):
            hit["rerank_score"] = float(score)
        hits.sort(key=lambda h: h.get("rerank_score", h.get("score", 0.0)), reverse=True)
        return hits

    def _encode_dense(self, text: str) -> List[float]:
        if self.dense_model is None:
            raise RuntimeError("Dense model not loaded but HNSW retrieval was requested.")

        if hasattr(self.dense_model, "encode"):
            embedding = self.dense_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        if self.dense_tokenizer is None:
            raise RuntimeError("Dense tokenizer missing for dense_model.")

        inputs = self.dense_tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()[0]
        return embeddings.tolist()

    def _encode_splade(self, text: str, top_k: int = 100) -> Dict[str, float]:
        if self.splade_model is None or self.splade_tokenizer is None:
            raise RuntimeError("SPLADE model/tokenizer not loaded but SPLADE retrieval was requested.")

        inputs = self.splade_tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.splade_model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            vec = torch.log1p(torch.relu(logits))
            vec = torch.max(vec, dim=1).values.squeeze()

        vec_cpu = vec.cpu().numpy()
        top_indices = np.argsort(vec_cpu)[-top_k:]

        sparse_dict = {}
        for idx in reversed(top_indices):
            score = float(vec_cpu[idx])
            if score <= 0:
                continue
            token = self.splade_tokenizer.convert_ids_to_tokens(int(idx))
            if not token or token.startswith("[") or token.startswith("<"):
                continue
            token_clean = token.replace("#", "").replace(".", "").replace(",", "")
            token_clean = "".join(c for c in token_clean if c.isalnum() or c in ["_", "-"])
            if token_clean:
                sparse_dict[token_clean] = score
        return sparse_dict

    def _retrieve_hnsw(
        self,
        corpus_name: str,
        query_text: str,
        max_buffer_count: int,
        max_hits_count: int,
        apply_reranker: bool = True,
    ) -> List[Dict]:
        query_vector = self._encode_dense(query_text)
        query = {
            "size": max_buffer_count,
            "knn": {
                "field": "dense_embedding",
                "query_vector": query_vector,
                "k": max_buffer_count,
                "num_candidates": max(50, max_buffer_count * 10),
            },
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
        }
        result = self._es.search(index=corpus_name, body=query)
        hits = self._dedupe_and_format_hits(result, max_hits_count)
        if apply_reranker:
            hits = self._apply_reranker(query_text, hits)
        return hits

    def _retrieve_splade(
        self,
        corpus_name: str,
        query_text: str,
        max_buffer_count: int,
        max_hits_count: int,
        apply_reranker: bool = True,
    ) -> List[Dict]:
        splade_dict = self._encode_splade(query_text)
        if not splade_dict:
            return []
        should_clauses = [
            {
                "term": {
                    f"splade_vector.{token}": {
                        "value": 1.0,
                        "boost": score,
                    }
                }
            }
            for token, score in splade_dict.items()
        ]
        query = {
            "size": max_buffer_count,
            "query": {
                "bool": {
                    "should": should_clauses,
                }
            },
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
        }
        result = self._es.search(index=corpus_name, body=query)
        hits = self._dedupe_and_format_hits(result, max_hits_count)
        if apply_reranker:
            hits = self._apply_reranker(query_text, hits)
        return hits

    def _retrieve_hybrid(
        self,
        corpus_name: str,
        query_text: str,
        max_buffer_count: int,
        max_hits_count: int,
        weights: Dict[str, float],
    ) -> List[Dict]:
        combined: Dict[str, Dict] = {}

        def add_hits(hits: List[Dict], weight: float):
            for hit in hits:
                doc_id = hit.get("es_doc_id") or hit.get("id")
                if doc_id is None:
                    continue
                if doc_id not in combined:
                    combined[doc_id] = {"source": hit.copy(), "score": 0.0}
                combined[doc_id]["score"] += weight * hit.get("score", 0.0)

        bm25_hits = self._retrieve_bm25(
            corpus_name=corpus_name,
            query_text=query_text,
            is_abstract=None,
            allowed_titles=None,
            allowed_paragraph_types=None,
            query_title_field_too=False,
            paragraph_index=None,
            max_buffer_count=max_buffer_count,
            max_hits_count=max_buffer_count,
            apply_reranker=False,
        )
        add_hits(bm25_hits, weights.get("bm25", 1.0))

        if self.dense_model is not None:
            hnsw_hits = self._retrieve_hnsw(
                corpus_name=corpus_name,
                query_text=query_text,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_buffer_count,
                apply_reranker=False,
            )
            add_hits(hnsw_hits, weights.get("dense", 1.0))

        if self.splade_model is not None:
            splade_hits = self._retrieve_splade(
                corpus_name=corpus_name,
                query_text=query_text,
                max_buffer_count=max_buffer_count,
                max_hits_count=max_buffer_count,
                apply_reranker=False,
            )
            add_hits(splade_hits, weights.get("splade", 1.0))

        merged = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        top = []
        top_hits = []
        for item in merged[:max_hits_count]:
            source = item["source"]
            source["score"] = item["score"]
            top_hits.append(source)

        top_hits = self._apply_reranker(query_text, top_hits)
        return top_hits


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="retrieve paragraphs or titles")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", choices={"hotpotqa", "2wikimultihopqa", "iirc", "musique"}
    )
    parser.add_argument("--host", type=str, help="host", default="localhost")
    parser.add_argument("--port", type=int, help="port", default=9200)
    args = parser.parse_args()

    retriever = ElasticsearchRetriever(
        corpus_name=args.dataset_name,
        host=args.host,
        port=args.port,
    )

    print("\n\nRetrieving Titles ...")
    results = retriever.retrieve_titles("injuries")
    for result in results:
        print(result)

    print("\n\nRetrieving Paragraphs ...")
    results = retriever.retrieve_paragraphs("injuries")
    for result in results:
        print(result)

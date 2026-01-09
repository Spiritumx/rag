from typing import List, Dict

# INNOVATION 1: Import v2 retriever
from elasticsearch_retriever_v2 import ElasticsearchRetriever


class UnifiedRetriever:
    """
    V2 Unified Retriever wrapper for ElasticsearchRetriever with Adaptive Retrieval.

    This class wrapper multiple different retrievers we experimented with.
    Since we settled with Elasticsearch, I've removed code related to other
    retrievers from here. Still keeping the wrapper for reproducibility.
    """

    def __init__(
        self,
        host: str = "http://localhost/",
        port: int = 9200,
        dense_model=None,
        dense_tokenizer=None,
        splade_model=None,
        splade_tokenizer=None,
        device: str = None,
        hybrid_weights=None,
        reranker=None,
        rerank_top_k: int = 50,
    ):
        self._elasticsearch_retriever = ElasticsearchRetriever(
            host=host,
            port=port,
            dense_model=dense_model,
            dense_tokenizer=dense_tokenizer,
            splade_model=splade_model,
            splade_tokenizer=splade_tokenizer,
            device=device,
            hybrid_weights=hybrid_weights,
            reranker=reranker,
            rerank_top_k=rerank_top_k,
        )

    def retrieve_from_elasticsearch(
        self,
        query_text: str,
        max_hits_count: int = 3,
        max_buffer_count: int = 100,
        document_type: str = "paragraph_text",
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        paragraph_index: int = None,
        corpus_name: str = None,
        retrieval_backend: str = "bm25",
        hybrid_weights=None,
        rerank_query_text: str = None,  # 新增：独立的rerank query
    ) -> List[Dict]:

        assert document_type in ("title", "paragraph_text", "title_paragraph_text")

        if paragraph_index is not None:
            assert (
                document_type == "paragraph_text"
            ), "paragraph_index not valid input for the document_type of paragraph_text."

        if self._elasticsearch_retriever is None:
            raise Exception("Elasticsearch retriever not initialized.")

        if document_type in ("paragraph_text", "title_paragraph_text"):
            is_abstract = True if corpus_name == "hotpotqa" else None  # Note "None" and not False
            query_title_field_too = document_type == "title_paragraph_text"
            paragraphs_results = self._elasticsearch_retriever.retrieve_paragraphs(
                query_text=query_text,
                is_abstract=is_abstract,
                max_hits_count=max_hits_count,
                allowed_titles=allowed_titles,
                allowed_paragraph_types=allowed_paragraph_types,
                paragraph_index=paragraph_index,
                corpus_name=corpus_name,
                query_title_field_too=query_title_field_too,
                max_buffer_count=max_buffer_count,
                retrieval_backend=retrieval_backend,
                hybrid_weights=hybrid_weights,
                rerank_query_text=rerank_query_text,  # 传递rerank query
            )

        elif document_type == "title":
            paragraphs_results = self._elasticsearch_retriever.retrieve_titles(
                query_text=query_text, max_hits_count=max_hits_count, corpus_name=corpus_name
            )

        return paragraphs_results

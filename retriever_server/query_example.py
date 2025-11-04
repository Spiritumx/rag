"""
Example script for querying different index types (BM25, HNSW, SPLADE)
"""
import argparse
from elasticsearch import Elasticsearch
import torch
from typing import List, Dict


def query_bm25(es: Elasticsearch, index_name: str, query_text: str, top_k: int = 10) -> List[Dict]:
    """Query using BM25 (text matching)"""
    print(f"\n=== BM25 Query ===")
    query = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^2", "paragraph_text"],  # title has 2x weight
                "type": "best_fields"
            }
        }
    }
    
    response = es.search(index=index_name, body=query)
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'score': hit['_score'],
            'title': hit['_source']['title'],
            'text': hit['_source']['paragraph_text'][:200] + '...'
        })
    return results


def query_hnsw(es: Elasticsearch, index_name: str, query_text: str, dense_model, top_k: int = 10) -> List[Dict]:
    """Query using HNSW dense vectors"""
    print(f"\n=== HNSW Dense Vector Query ===")
    
    # Generate query vector
    query_vector = dense_model.encode(query_text, convert_to_numpy=True).tolist()
    
    # Use kNN query (ES 8.0+)
    query = {
        "size": top_k,
        "knn": {
            "field": "dense_embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10  # Search more candidates for better recall
        },
        "_source": ["title", "paragraph_text"]
    }
    
    try:
        response = es.search(index=index_name, body=query)
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'score': hit['_score'],
                'title': hit['_source']['title'],
                'text': hit['_source']['paragraph_text'][:200] + '...'
            })
        return results
    except Exception as e:
        print(f"Error: {e}")
        print("Trying alternative query method...")
        
        # Fallback to script_score query
        query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'dense_embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "_source": ["title", "paragraph_text"]
        }
        response = es.search(index=index_name, body=query)
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'score': hit['_score'],
                'title': hit['_source']['title'],
                'text': hit['_source']['paragraph_text'][:200] + '...'
            })
        return results


def query_splade(es: Elasticsearch, index_name: str, query_text: str, splade_model, splade_tokenizer, device, top_k: int = 10) -> List[Dict]:
    """Query using SPLADE sparse vectors"""
    print(f"\n=== SPLADE Sparse Vector Query ===")
    
    # Generate SPLADE vector for query
    inputs = splade_tokenizer(query_text, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = splade_model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        vec = torch.log(1 + torch.relu(logits))
        vec = torch.max(vec, dim=1).values.squeeze()
    
    # Convert to sparse dict (keep top tokens)
    vec_cpu = vec.cpu().numpy()
    top_k_tokens = 100  # Keep top 100 query tokens
    top_indices = vec_cpu.argsort()[-top_k_tokens:][::-1]
    
    splade_dict = {}
    for idx in top_indices:
        if vec_cpu[idx] > 0:
            token = splade_tokenizer.convert_ids_to_tokens([idx])[0]
            splade_dict[token] = float(vec_cpu[idx])
    
    print(f"Query expansion: {len(splade_dict)} tokens")
    print(f"Top tokens: {list(splade_dict.keys())[:10]}")
    
    # Build bool query with rank_feature
    should_clauses = [
        {
            "rank_feature": {
                "field": "splade_vector",
                "boost": score,
                "linear": {}
            },
            "term": {
                "splade_vector": token
            }
        }
        for token, score in splade_dict.items()
    ]
    
    query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": should_clauses
            }
        },
        "_source": ["title", "paragraph_text"]
    }
    
    response = es.search(index=index_name, body=query)
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'score': hit['_score'],
            'title': hit['_source']['title'],
            'text': hit['_source']['paragraph_text'][:200] + '...'
        })
    return results


def query_hybrid(es: Elasticsearch, index_name: str, query_text: str, dense_model, 
                 bm25_weight: float = 1.0, dense_weight: float = 1.0, top_k: int = 10) -> List[Dict]:
    """Hybrid query combining BM25 and dense vectors"""
    print(f"\n=== Hybrid Query (BM25 + HNSW) ===")
    print(f"Weights - BM25: {bm25_weight}, Dense: {dense_weight}")
    
    # Generate query vector
    query_vector = dense_model.encode(query_text, convert_to_numpy=True).tolist()
    
    query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title^2", "paragraph_text"]
                    }
                },
                "script": {
                    "source": f"({bm25_weight} * _score) + ({dense_weight} * (cosineSimilarity(params.query_vector, 'dense_embedding') + 1.0))",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "_source": ["title", "paragraph_text"]
    }
    
    response = es.search(index=index_name, body=query)
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'score': hit['_score'],
            'title': hit['_source']['title'],
            'text': hit['_source']['paragraph_text'][:200] + '...'
        })
    return results


def print_results(results: List[Dict], method: str):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"{method} - Top {len(results)} Results")
    print('='*80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Title: {result['title']}")
        print(f"   Text: {result['text']}")


def main():
    parser = argparse.ArgumentParser(description="Query different index types")
    parser.add_argument("index_name", help="Name of the Elasticsearch index")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--method", choices=["bm25", "hnsw", "splade", "hybrid", "all"], 
                       default="all", help="Query method to use")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Dense embedding model")
    parser.add_argument("--splade-model", default="naver/splade-v3",
                       help="SPLADE model")
    parser.add_argument("--es-host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch port")
    
    args = parser.parse_args()
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        [{"host": args.es_host, "port": args.es_port, "scheme": "http"}],
        request_timeout=30
    )
    
    # Check if index exists
    if not es.indices.exists(index=args.index_name):
        print(f"Error: Index '{args.index_name}' does not exist")
        return
    
    # Get index mappings to see what's available
    mappings = es.indices.get_mapping(index=args.index_name)
    properties = mappings[args.index_name]['mappings']['properties']
    has_dense = 'dense_embedding' in properties
    has_splade = 'splade_vector' in properties
    
    print(f"Index: {args.index_name}")
    print(f"Available index types:")
    print(f"  - BM25: Yes (always available)")
    print(f"  - HNSW: {'Yes' if has_dense else 'No'}")
    print(f"  - SPLADE: {'Yes' if has_splade else 'No'}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load models as needed
    dense_model = None
    splade_model = None
    splade_tokenizer = None
    
    if args.method in ["hnsw", "hybrid", "all"] and has_dense:
        print(f"\nLoading dense model: {args.dense_model}")
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer(args.dense_model)
        dense_model = dense_model.to(device)
    
    if args.method in ["splade", "all"] and has_splade:
        print(f"\nLoading SPLADE model: {args.splade_model}")
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        splade_tokenizer = AutoTokenizer.from_pretrained(args.splade_model)
        splade_model = AutoModelForMaskedLM.from_pretrained(args.splade_model)
        splade_model = splade_model.to(device)
        splade_model.eval()
    
    print(f"\nQuery: '{args.query}'")
    
    # Execute queries
    if args.method in ["bm25", "all"]:
        results = query_bm25(es, args.index_name, args.query, args.top_k)
        print_results(results, "BM25")
    
    if args.method in ["hnsw", "all"] and has_dense and dense_model:
        results = query_hnsw(es, args.index_name, args.query, dense_model, args.top_k)
        print_results(results, "HNSW")
    
    if args.method in ["splade", "all"] and has_splade and splade_model:
        results = query_splade(es, args.index_name, args.query, splade_model, splade_tokenizer, device, args.top_k)
        print_results(results, "SPLADE")
    
    if args.method in ["hybrid", "all"] and has_dense and dense_model:
        results = query_hybrid(es, args.index_name, args.query, dense_model, 
                             bm25_weight=1.0, dense_weight=2.0, top_k=args.top_k)
        print_results(results, "Hybrid")


if __name__ == "__main__":
    main()


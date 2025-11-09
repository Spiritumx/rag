""" 
Build ES (Elasticsearch) Multi-Index (BM25 + HNSW + SPLADE).
"""
from typing import Dict, Union, Optional, Any
import json
import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import hashlib
import io
import dill
from tqdm import tqdm
import glob
import bz2
import base58
from bs4 import BeautifulSoup
import os
import random
import csv
import torch
import numpy as np


def hash_object(o: Any) -> str:
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def generate_dense_embedding(text: str, model, tokenizer, device, max_length: int = 512):
    """Generate dense embeddings using sentence transformers or similar models."""
    if model is None:
        return None
    
    try:
        # For sentence-transformers models
        if hasattr(model, 'encode'):
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # For HuggingFace transformers models
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, 
                             truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy()[0]
            return embeddings.tolist()
    except Exception as e:
        print(f"Error generating dense embedding: {e}")
        return None


def generate_dense_embeddings_batch(texts: list, model, tokenizer, device, max_length: int = 512, batch_size: int = 32):
    """Generate dense embeddings for a batch of texts."""
    if model is None or not texts:
        return [None] * len(texts)
    
    try:
        # For sentence-transformers models
        if hasattr(model, 'encode'):
            embeddings = model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        else:
            # For HuggingFace transformers models - process in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, 
                                 truncation=True, padding=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = embeddings.cpu().numpy()
                all_embeddings.extend([emb.tolist() for emb in embeddings])
            return all_embeddings
    except Exception as e:
        print(f"Error generating dense embeddings batch: {e}")
        return [None] * len(texts)


def generate_splade_vector(text: str, model, tokenizer, device, max_length: int = 512):
    """Generate SPLADE sparse vectors."""
    if model is None:
        return None
    
    try:
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, 
                          truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # SPLADE uses log(1 + ReLU(logits)) and max pooling
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            vec = torch.log(1 + torch.relu(logits))
            vec = torch.max(vec, dim=1).values.squeeze()
            
            # Convert to sparse representation (keep only non-zero values)
            vec_cpu = vec.cpu().numpy()
            # Keep top-k values to make it sparse (e.g., top 100-200 tokens)
            top_k = 200
            top_indices = np.argsort(vec_cpu)[-top_k:]
            
            # Create sparse dict: {token: weight}
            sparse_dict = {}
            for idx in top_indices:
                if vec_cpu[idx] > 0:
                    # Convert token_id to token string for ES rank_features
                    token = tokenizer.convert_ids_to_tokens([int(idx)])[0]
                    
                    # Filter out special tokens and invalid tokens
                    # rank_features doesn't support: [, <, ., and some other special chars
                    if token and not token.startswith('[') and not token.startswith('<') and '.' not in token:
                        # Clean token: replace special chars that ES doesn't like
                        token_clean = token.replace('#', '').replace('##', '').replace('.', '').replace(',', '')
                        # Remove any remaining special characters that could cause issues
                        token_clean = ''.join(c for c in token_clean if c.isalnum() or c in ['_', '-'])
                        if token_clean and len(token_clean) > 0:  # Make sure it's not empty after cleaning
                            sparse_dict[token_clean] = float(vec_cpu[idx])
            
            return sparse_dict if sparse_dict else None
    except Exception as e:
        print(f"Error generating SPLADE vector: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_splade_vectors_batch(texts: list, model, tokenizer, device, max_length: int = 512, batch_size: int = 32):
    """Generate SPLADE sparse vectors for a batch of texts."""
    if model is None or not texts:
        return [None] * len(texts)
    
    try:
        all_sparse_dicts = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, 
                              truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # SPLADE uses log(1 + ReLU(logits)) and max pooling
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                vecs = torch.log(1 + torch.relu(logits))
                vecs = torch.max(vecs, dim=1).values  # [batch_size, vocab_size]
                
                vecs_cpu = vecs.cpu().numpy()
                
                # Process each vector in the batch
                for vec_cpu in vecs_cpu:
                    top_k = 200
                    top_indices = np.argsort(vec_cpu)[-top_k:]
                    
                    sparse_dict = {}
                    for idx in top_indices:
                        if vec_cpu[idx] > 0:
                            token = tokenizer.convert_ids_to_tokens([int(idx)])[0]
                            if token and not token.startswith('[') and not token.startswith('<') and '.' not in token:
                                token_clean = token.replace('#', '').replace('##', '').replace('.', '').replace(',', '')
                                token_clean = ''.join(c for c in token_clean if c.isalnum() or c in ['_', '-'])
                                if token_clean and len(token_clean) > 0:
                                    sparse_dict[token_clean] = float(vec_cpu[idx])
                    
                    all_sparse_dicts.append(sparse_dict if sparse_dict else None)
        
        return all_sparse_dicts
    except Exception as e:
        print(f"Error generating SPLADE vectors batch: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(texts)


def make_hotpotqa_documents(
    elasticsearch_index: str, 
    metadata: Union[Dict[str, int], None] = None,
    dense_model=None,
    dense_tokenizer=None,
    splade_model=None,
    splade_tokenizer=None,
    device=None
):
    raw_glob_filepath = os.path.join("/root/autodl-tmp/raw_data", "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2")
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata
    for filepath in tqdm(glob.glob(raw_glob_filepath)):
        for datum in bz2.BZ2File(filepath).readlines():
            instance = json.loads(datum.strip())

            id_ = hash_object(instance)[:32]
            title = instance["title"]
            sentences_text = [e.strip() for e in instance["text"]]
            paragraph_text = " ".join(sentences_text)
            url = instance["url"]
            is_abstract = True
            paragraph_index = 0

            es_paragraph = {
                "id": id_,
                "title": title,
                "paragraph_index": paragraph_index,
                "paragraph_text": paragraph_text,
                "url": url,
                "is_abstract": is_abstract,
            }
            
            # Generate embeddings if models are provided
            text_to_embed = f"{title} {paragraph_text}"
            
            if dense_model is not None:
                dense_emb = generate_dense_embedding(text_to_embed, dense_model, dense_tokenizer, device)
                if dense_emb is not None:
                    es_paragraph["dense_embedding"] = dense_emb
            
            if splade_model is not None:
                splade_vec = generate_splade_vector(text_to_embed, splade_model, splade_tokenizer, device)
                if splade_vec is not None:
                    es_paragraph["splade_vector"] = splade_vec
            
            document = {
                "_op_type": "create",
                "_index": elasticsearch_index,
                "_id": metadata["idx"],
                "_source": es_paragraph,
            }
            yield (document)
            metadata["idx"] += 1


def make_iirc_documents(
    elasticsearch_index: str, 
    metadata: Union[Dict[str, int], None] = None,
    dense_model=None,
    dense_tokenizer=None,
    splade_model=None,
    splade_tokenizer=None,
    device=None
):
    raw_filepath = os.path.join("/root/autodl-tmp/raw_data", "iirc", "context_articles.json")

    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    random.seed(13370)  # Don't change.

    with open(raw_filepath, "r") as file:
        full_data = json.load(file)

        for title, page_html in tqdm(full_data.items()):
            page_soup = BeautifulSoup(page_html, "html.parser")
            paragraph_texts = [
                text for text in page_soup.text.split("\n") if text.strip() and len(text.strip().split()) > 10
            ]

            # IIRC has a positional bias. 70% of the times, the first
            # is the supporting one, and almost all are in 1st 20.
            # So we scramble them to make it more challenging retrieval
            # problem.
            paragraph_indices_and_texts = [
                (paragraph_index, paragraph_text) for paragraph_index, paragraph_text in enumerate(paragraph_texts)
            ]
            random.shuffle(paragraph_indices_and_texts)
            for paragraph_index, paragraph_text in paragraph_indices_and_texts:
                url = ""
                id_ = hash_object(title + paragraph_text)
                is_abstract = paragraph_index == 0
                es_paragraph = {
                    "id": id_,
                    "title": title,
                    "paragraph_index": paragraph_index,
                    "paragraph_text": paragraph_text,
                    "url": url,
                    "is_abstract": is_abstract,
                }
                
                # Generate embeddings if models are provided
                text_to_embed = f"{title} {paragraph_text}"
                
                if dense_model is not None:
                    dense_emb = generate_dense_embedding(text_to_embed, dense_model, dense_tokenizer, device)
                    if dense_emb is not None:
                        es_paragraph["dense_embedding"] = dense_emb
                
                if splade_model is not None:
                    splade_vec = generate_splade_vector(text_to_embed, splade_model, splade_tokenizer, device)
                    if splade_vec is not None:
                        es_paragraph["splade_vector"] = splade_vec
                
                document = {
                    "_op_type": "create",
                    "_index": elasticsearch_index,
                    "_id": metadata["idx"],
                    "_source": es_paragraph,
                }
                yield (document)
                metadata["idx"] += 1


def make_2wikimultihopqa_documents(
    elasticsearch_index: str, 
    metadata: Union[Dict[str, int], None] = None,
    dense_model=None,
    dense_tokenizer=None,
    splade_model=None,
    splade_tokenizer=None,
    device=None
):
    raw_filepaths = [
        os.path.join("/root/autodl-tmp/raw_data", "2wikimultihopqa", "train.json"),
        os.path.join("/root/autodl-tmp/raw_data", "2wikimultihopqa", "dev.json"),
        os.path.join("/root/autodl-tmp/raw_data", "2wikimultihopqa", "test.json"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:

        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):

                for paragraph in instance["context"]:

                    title = paragraph[0]
                    paragraph_text = " ".join(paragraph[1])
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                    }
                    
                    # Generate embeddings if models are provided
                    text_to_embed = f"{title} {paragraph_text}"
                    
                    if dense_model is not None:
                        dense_emb = generate_dense_embedding(text_to_embed, dense_model, dense_tokenizer, device)
                        if dense_emb is not None:
                            es_paragraph["dense_embedding"] = dense_emb
                    
                    if splade_model is not None:
                        splade_vec = generate_splade_vector(text_to_embed, splade_model, splade_tokenizer, device)
                        if splade_vec is not None:
                            es_paragraph["splade_vector"] = splade_vec
                    
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1


def make_musique_documents(
    elasticsearch_index: str, 
    metadata: Union[Dict[str, int], None] = None,
    dense_model=None,
    dense_tokenizer=None,
    splade_model=None,
    splade_tokenizer=None,
    device=None
):
    raw_filepaths = [
        os.path.join("/root/autodl-tmp/raw_data", "musique", "musique_ans_v1.0_dev.jsonl"),
        os.path.join("/root/autodl-tmp/raw_data", "musique", "musique_ans_v1.0_test.jsonl"),
        os.path.join("/root/autodl-tmp/raw_data", "musique", "musique_ans_v1.0_train.jsonl"),
        os.path.join("/root/autodl-tmp/raw_data", "musique", "musique_full_v1.0_dev.jsonl"),
        os.path.join("/root/autodl-tmp/raw_data", "musique", "musique_full_v1.0_test.jsonl"),
        os.path.join("/root/autodl-tmp/raw_data", "musique", "musique_full_v1.0_train.jsonl"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:

        with open(raw_filepath, "r") as file:
            for line in tqdm(file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line)

                for paragraph in instance["paragraphs"]:

                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                    }
                    
                    # Generate embeddings if models are provided
                    text_to_embed = f"{title} {paragraph_text}"
                    
                    if dense_model is not None:
                        dense_emb = generate_dense_embedding(text_to_embed, dense_model, dense_tokenizer, device)
                        if dense_emb is not None:
                            es_paragraph["dense_embedding"] = dense_emb
                    
                    if splade_model is not None:
                        splade_vec = generate_splade_vector(text_to_embed, splade_model, splade_tokenizer, device)
                        if splade_vec is not None:
                            es_paragraph["splade_vector"] = splade_vec
                    
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1

def make_wiki_documents(
    elasticsearch_index: str, 
    metadata: Union[Dict[str, int], None] = None,
    dense_model=None,
    dense_tokenizer=None,
    splade_model=None,
    splade_tokenizer=None,
    device=None,
    batch_size: int = 32
):
    raw_glob_filepath = os.path.join("/root/autodl-tmp/raw_data", "wiki", 'psgs_w100.tsv')
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    with open(raw_glob_filepath) as input_file:
        tr = csv.reader(input_file, delimiter='\t')
        next(tr)
        
        # Batch processing
        batch_data = []
        batch_texts = []
        
        for line in tqdm(tr):
            paragraph_text = line[1]
            title = line[2]
            url = ""
            
            id_ = hash_object(" ".join([title, paragraph_text]))[:32]
            paragraph_index = 0
            is_abstract = True

            es_paragraph = {
                "id": id_,
                "title": title,
                "paragraph_index": paragraph_index,
                "paragraph_text": paragraph_text,
                "url": url,
                "is_abstract": is_abstract,
            }
            
            text_to_embed = f"{title} {paragraph_text}"
            batch_data.append(es_paragraph)
            batch_texts.append(text_to_embed)
            
            # Process batch when it reaches batch_size
            if len(batch_data) >= batch_size:
                # Generate embeddings in batch
                if dense_model is not None:
                    dense_embs = generate_dense_embeddings_batch(batch_texts, dense_model, dense_tokenizer, device, batch_size=batch_size)
                    for i, dense_emb in enumerate(dense_embs):
                        if dense_emb is not None:
                            batch_data[i]["dense_embedding"] = dense_emb
                
                if splade_model is not None:
                    splade_vecs = generate_splade_vectors_batch(batch_texts, splade_model, splade_tokenizer, device, batch_size=batch_size)
                    for i, splade_vec in enumerate(splade_vecs):
                        if splade_vec is not None:
                            batch_data[i]["splade_vector"] = splade_vec
                
                # Yield documents
                for es_paragraph in batch_data:
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield document
                    metadata["idx"] += 1
                
                # Clear batch
                batch_data = []
                batch_texts = []
        
        # Process remaining batch
        if batch_data:
            if dense_model is not None:
                dense_embs = generate_dense_embeddings_batch(batch_texts, dense_model, dense_tokenizer, device, batch_size=batch_size)
                for i, dense_emb in enumerate(dense_embs):
                    if dense_emb is not None:
                        batch_data[i]["dense_embedding"] = dense_emb
            
            if splade_model is not None:
                splade_vecs = generate_splade_vectors_batch(batch_texts, splade_model, splade_tokenizer, device, batch_size=batch_size)
                for i, splade_vec in enumerate(splade_vecs):
                    if splade_vec is not None:
                        batch_data[i]["splade_vector"] = splade_vec
            
            for es_paragraph in batch_data:
                document = {
                    "_op_type": "create",
                    "_index": elasticsearch_index,
                    "_id": metadata["idx"],
                    "_source": es_paragraph,
                }
                yield document
                metadata["idx"] += 1




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Index paragraphs in Elasticsearch with BM25, HNSW, and SPLADE")
    parser.add_argument(
        "dataset_name",
        help="name of the dataset",
        type=str,
        choices=("hotpotqa", "iirc", "2wikimultihopqa", "musique", 'nq', 'wiki', 'trivia', 'squad'),
    )
    parser.add_argument(
        "--force",
        help="force delete before creating new index.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-dense",
        help="Generate and index dense embeddings for HNSW search",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dense-model",
        help="Dense embedding model name (sentence-transformers or HuggingFace)",
        type=str,
        default="all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--dense-model-path",
        help="Path to local dense model directory (overrides --dense-model)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use-splade",
        help="Generate and index SPLADE sparse vectors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--splade-model",
        help="SPLADE model name",
        type=str,
        default="naver/splade-v3",
    )
    parser.add_argument(
        "--splade-model-path",
        help="Path to local SPLADE model directory (overrides --splade-model)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for embedding generation",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--es-chunk-size",
        help="Elasticsearch bulk chunk size (documents per batch)",
        type=int,
        default=2000,
    )
    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Auto-detect local models in retriever_server/models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.join(script_dir, "models")
    
    # If no path specified, check local directory for models
    if args.use_dense and not args.dense_model_path:
        local_dense_path = os.path.join(local_model_dir, "dense", "all-MiniLM-L6-v2")
        if os.path.exists(local_dense_path):
            args.dense_model_path = local_dense_path
            print(f"📁 Found local dense model: {local_dense_path}")
    
    if args.use_splade and not args.splade_model_path:
        local_splade_path = os.path.join(local_model_dir, "splade", "splade-v3")
        if os.path.exists(local_splade_path):
            args.splade_model_path = local_splade_path
            print(f"📁 Found local SPLADE model: {local_splade_path}")

    # Load models based on arguments
    dense_model = None
    dense_tokenizer = None
    splade_model = None
    splade_tokenizer = None

    if args.use_dense:
        # Use local path if provided, otherwise use model name
        model_source = args.dense_model_path if args.dense_model_path else args.dense_model
        print(f"Loading dense embedding model: {model_source}")
        if args.dense_model_path:
            print(f"  ✓ Using local model")
        
        try:
            # Try sentence-transformers first
            from sentence_transformers import SentenceTransformer
            dense_model = SentenceTransformer(model_source)
            dense_model = dense_model.to(device)
            print(f"✓ Dense model loaded successfully (dimension: {dense_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            print(f"Failed to load as sentence-transformer, trying HuggingFace: {e}")
            try:
                from transformers import AutoModel, AutoTokenizer
                dense_tokenizer = AutoTokenizer.from_pretrained(model_source)
                dense_model = AutoModel.from_pretrained(model_source)
                dense_model = dense_model.to(device)
                dense_model.eval()
                print(f"✓ Dense model loaded successfully via HuggingFace")
            except Exception as e2:
                print(f"✗ Failed to load dense model: {e2}")
                args.use_dense = False

    if args.use_splade:
        # Use local path if provided, otherwise use model name
        model_source = args.splade_model_path if args.splade_model_path else args.splade_model
        print(f"Loading SPLADE model: {model_source}")
        if args.splade_model_path:
            print(f"  ✓ Using local model")
        
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            splade_tokenizer = AutoTokenizer.from_pretrained(model_source)
            splade_model = AutoModelForMaskedLM.from_pretrained(model_source)
            splade_model = splade_model.to(device)
            splade_model.eval()
            print(f"✓ SPLADE model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load SPLADE model: {e}")
            args.use_splade = False

    # conntect elastic-search
    elastic_host = "localhost"
    elastic_port = 9200
    elasticsearch_index = args.dataset_name
    es = Elasticsearch(
        [{"host": elastic_host, "port": elastic_port, "scheme": "http"}],
        max_retries=20,  # it's exp backoff starting 2, more than 2 retries will be too much.
        request_timeout=2000,
        retry_on_timeout=True,
    )

    # Base mappings (BM25)
    paragraphs_index_settings = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "english",
                },
                "paragraph_index": {"type": "integer"},
                "paragraph_text": {
                    "type": "text",
                    "analyzer": "english",
                },
                "url": {
                    "type": "text",
                    "analyzer": "english",
                },
                "is_abstract": {"type": "boolean"},
            }
        }
    }

    # Add dense vector mapping if using dense embeddings (HNSW)
    if args.use_dense:
        if hasattr(dense_model, 'get_sentence_embedding_dimension'):
            dense_dim = dense_model.get_sentence_embedding_dimension()
        else:
            # Default dimension for common models
            dense_dim = 768
        
        paragraphs_index_settings["mappings"]["properties"]["dense_embedding"] = {
            "type": "dense_vector",
            "dims": dense_dim,
            "index": True,
            "similarity": "cosine",  # or "dot_product", "l2_norm"
            "index_options": {
                "type": "hnsw",
                "m": 16,  # Number of connections per node
                "ef_construction": 100  # Size of candidate list during construction
            }
        }
        print(f"Added dense_embedding field with dimension {dense_dim} and HNSW index")

    # Add SPLADE sparse vector mapping if using SPLADE
    if args.use_splade:
        paragraphs_index_settings["mappings"]["properties"]["splade_vector"] = {
            "type": "rank_features"  # Suitable for sparse vectors with term:score pairs
        }
        print(f"Added splade_vector field with rank_features type")

    # Check if index exists - using try/except for compatibility
    try:
        index_exists = es.indices.exists(index=elasticsearch_index)
    except:
        # Fallback for newer ES client versions
        from elasticsearch import NotFoundError
        try:
            es.indices.get(index=elasticsearch_index)
            index_exists = True
        except NotFoundError:
            index_exists = False
    
    print("Index already exists" if index_exists else "Index doesn't exist.")

    # delete index if exists
    if index_exists:

        if not args.force:
            feedback = input(f"Index {elasticsearch_index} already exists. " f"Are you sure you want to delete it?")
            if not (feedback.startswith("y") or feedback == ""):
                exit("Termited by user.")
        es.indices.delete(index=elasticsearch_index)

    # create index
    print("Creating Index ...")
    es.indices.create(index=elasticsearch_index, mappings=paragraphs_index_settings["mappings"])

    if args.dataset_name == "hotpotqa":
        make_documents_func = make_hotpotqa_documents
    elif args.dataset_name == "iirc":
        make_documents_func = make_iirc_documents
    elif args.dataset_name == "2wikimultihopqa":
        make_documents_func = make_2wikimultihopqa_documents
    elif args.dataset_name == "musique":
        make_documents_func = make_musique_documents
    elif args.dataset_name == "wiki":
        make_documents_func = make_wiki_documents 
    else:
        raise Exception(f"Unknown dataset_name {args.dataset_name}")

    # Bulk-insert documents into index
    print("Inserting Paragraphs ...")
    print(f"Using BM25: Yes")
    print(f"Using Dense Embeddings (HNSW): {args.use_dense}")
    print(f"Using SPLADE: {args.use_splade}")
    print(f"Batch size for embeddings: {args.batch_size}")
    print(f"ES bulk chunk size: {args.es_chunk_size}")
    
    # Prepare kwargs for make_documents_func
    doc_kwargs = {
        "dense_model": dense_model,
        "dense_tokenizer": dense_tokenizer,
        "splade_model": splade_model,
        "splade_tokenizer": splade_tokenizer,
        "device": device
    }
    
    # Add batch_size for wiki dataset
    if args.dataset_name == "wiki":
        doc_kwargs["batch_size"] = args.batch_size
    
    try:
        result = bulk(
            es,
            make_documents_func(elasticsearch_index, **doc_kwargs),
            chunk_size=args.es_chunk_size,
            raise_on_error=True,
            raise_on_exception=True,
            max_retries=2,
            request_timeout=500,
        )
        es.indices.refresh(index=elasticsearch_index)
        document_count = result[0]
        print(f"Index {elasticsearch_index} is ready. Added {document_count} documents.")
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Bulk indexing failed!")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        # Try to extract detailed error information
        if hasattr(e, 'errors'):
            print(f"\nNumber of failed documents: {len(e.errors)}")
            print("\nFirst few errors:")
            for i, error in enumerate(e.errors[:5]):  # Show first 5 errors
                print(f"\n--- Error {i+1} ---")
                print(json.dumps(error, indent=2))
        
        print("\n" + "="*80)
        print("Possible causes:")
        print("1. SPLADE vector format issue - rank_features expects dict with string keys")
        print("2. Dense embedding dimension mismatch")
        print("3. Missing or invalid field in some documents")
        print("4. Memory or resource limits in Elasticsearch")
        print("="*80)
        raise
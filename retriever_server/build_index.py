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
            
            # Create sparse dict: {token_id: weight}
            sparse_dict = {}
            for idx in top_indices:
                if vec_cpu[idx] > 0:
                    # Convert token_id to token string for ES rank_features
                    token = tokenizer.convert_ids_to_tokens([idx])[0]
                    sparse_dict[token] = float(vec_cpu[idx])
            
            return sparse_dict if sparse_dict else None
    except Exception as e:
        print(f"Error generating SPLADE vector: {e}")
        return None


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
    device=None
):
    raw_glob_filepath = os.path.join("/root/autodl-tmp/raw_data", "wiki", 'psgs_w100.tsv')
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    with open(raw_glob_filepath) as input_file:
        tr = csv.reader(input_file, delimiter='\t')
        next(tr)
        for line in tqdm(tr):
            #import pdb; pdb.set_trace()
            #dict_line['_id'] = line[0]
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
        default="sentence-transformers/all-MiniLM-L6-v2",
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
        "--batch-size",
        help="Batch size for embedding generation (future use)",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models based on arguments
    dense_model = None
    dense_tokenizer = None
    splade_model = None
    splade_tokenizer = None

    if args.use_dense:
        print(f"Loading dense embedding model: {args.dense_model}")
        try:
            # Try sentence-transformers first
            from sentence_transformers import SentenceTransformer
            dense_model = SentenceTransformer(args.dense_model)
            dense_model = dense_model.to(device)
            print(f"Dense model loaded successfully (dimension: {dense_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            print(f"Failed to load as sentence-transformer, trying HuggingFace: {e}")
            try:
                from transformers import AutoModel, AutoTokenizer
                dense_tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
                dense_model = AutoModel.from_pretrained(args.dense_model)
                dense_model = dense_model.to(device)
                dense_model.eval()
                print(f"Dense model loaded successfully via HuggingFace")
            except Exception as e2:
                print(f"Failed to load dense model: {e2}")
                args.use_dense = False

    if args.use_splade:
        print(f"Loading SPLADE model: {args.splade_model}")
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            splade_tokenizer = AutoTokenizer.from_pretrained(args.splade_model)
            splade_model = AutoModelForMaskedLM.from_pretrained(args.splade_model)
            splade_model = splade_model.to(device)
            splade_model.eval()
            print(f"SPLADE model loaded successfully")
        except Exception as e:
            print(f"Failed to load SPLADE model: {e}")
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
    
    result = bulk(
        es,
        make_documents_func(
            elasticsearch_index,
            dense_model=dense_model,
            dense_tokenizer=dense_tokenizer,
            splade_model=splade_model,
            splade_tokenizer=splade_tokenizer,
            device=device
        ),
        raise_on_error=True,  # set to true o/w it'll fail silently and only show less docs.
        raise_on_exception=True,  # set to true o/w it'll fail silently and only show less docs.
        max_retries=2,  # it's exp backoff starting 2, more than 2 retries will be too much.
        request_timeout=500,
    )
    es.indices.refresh(index=elasticsearch_index)  # actually updates the count.
    document_count = result[0]
    print(f"Index {elasticsearch_index} is ready. Added {document_count} documents.")
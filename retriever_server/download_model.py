"""
Download models to local directory for offline use.
"""
import os
import argparse
from pathlib import Path


def download_sentence_transformer(model_name: str, save_dir: str):
    """Download sentence-transformers model"""
    print(f"📥 Downloading sentence-transformers model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Create a cleaner path structure
        model_simple_name = model_name.split("/")[-1]
        local_path = os.path.join(save_dir, "dense", model_simple_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"   Loading model from HuggingFace...")
        model = SentenceTransformer(model_name)
        
        print(f"   Saving to local directory...")
        model.save(local_path)
        
        # Test loading from local path
        print(f"   Testing local model...")
        test_model = SentenceTransformer(local_path)
        dim = test_model.get_sentence_embedding_dimension()
        
        print(f"✓ Successfully saved to: {local_path}")
        print(f"  Model dimension: {dim}")
        return local_path
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_transformers_model(model_name: str, save_dir: str, model_type: str = "splade"):
    """Download HuggingFace transformers model"""
    print(f"📥 Downloading transformers model: {model_name}")
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        
        # Create a cleaner path structure
        model_simple_name = model_name.split("/")[-1]
        local_path = os.path.join(save_dir, model_type, model_simple_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        print(f"   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_path)
        
        print(f"   Downloading model weights...")
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.save_pretrained(local_path)
        
        # Test loading from local path
        print(f"   Testing local model...")
        test_tokenizer = AutoTokenizer.from_pretrained(local_path)
        test_model = AutoModelForMaskedLM.from_pretrained(local_path)
        
        print(f"✓ Successfully saved to: {local_path}")
        print(f"  Vocab size: {len(test_tokenizer)}")
        return local_path
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "models")
    
    parser = argparse.ArgumentParser(
        description="Download models for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models to default location (retriever_server/models)
  python download_model.py --all

  # Download only dense embedding model
  python download_model.py --download-dense

  # Download only SPLADE model
  python download_model.py --download-splade

  # Download specific models
  python download_model.py --download-dense --dense-model sentence-transformers/all-mpnet-base-v2
        """
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=default_model_dir,
        help=f"Directory to save models (default: {default_model_dir})"
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Dense embedding model to download (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--splade-model",
        type=str,
        default="naver/splade-cocondenser-ensembledistil",
        help="SPLADE model to download (default: splade-cocondenser-ensembledistil)"
    )
    parser.add_argument(
        "--download-dense",
        action="store_true",
        default=False,
        help="Download dense embedding model"
    )
    parser.add_argument(
        "--download-splade",
        action="store_true",
        default=False,
        help="Download SPLADE model"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Download all models"
    )
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    print("="*80)
    print("MODEL DOWNLOAD TOOL")
    print("="*80)
    print(f"📁 Model directory: {os.path.abspath(args.model_dir)}")
    print("="*80)
    
    downloaded = []
    
    # Download dense model
    if args.download_dense or args.all:
        print(f"\n[1/2] Dense Embedding Model")
        print("-"*80)
        path = download_sentence_transformer(args.dense_model, args.model_dir)
        if path:
            downloaded.append(("Dense", args.dense_model, path))
    
    # Download SPLADE model
    if args.download_splade or args.all:
        print(f"\n[2/2] SPLADE Model")
        print("-"*80)
        path = download_transformers_model(args.splade_model, args.model_dir, "splade")
        if path:
            downloaded.append(("SPLADE", args.splade_model, path))
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    
    if downloaded:
        for model_type, model_name, local_path in downloaded:
            print(f"✓ {model_type}: {model_name}")
            print(f"  → {local_path}")
    else:
        print("No models downloaded. Use --download-dense, --download-splade, or --all")
        print("\nExamples:")
        print("  python download_model.py --all")
        print("  python download_model.py --download-dense")
        print("  python download_model.py --download-splade")
    
    # Print usage instructions
    if downloaded:
        print("\n" + "="*80)
        print("✅ MODELS READY TO USE")
        print("="*80)
        print("Models have been downloaded to the default location.")
        print("build_index.py will automatically load them from there.")
        print()
        print("Usage:")
        
        # Build simple command
        cmd_parts = ["python retriever_server/build_index.py <dataset>"]
        
        has_dense = any(m[0] == "Dense" for m in downloaded)
        has_splade = any(m[0] == "SPLADE" for m in downloaded)
        
        if has_dense:
            cmd_parts.append("--use-dense")
        if has_splade:
            cmd_parts.append("--use-splade")
        
        print("  " + " ".join(cmd_parts))
        print()
        print("Example:")
        
        # Generate example command
        example_cmd = "python retriever_server/build_index.py wiki"
        if has_dense:
            example_cmd += " --use-dense"
        if has_splade:
            example_cmd += " --use-splade"
        
        print(f"  {example_cmd}")
        print()
        print("Note: Models are stored in:", args.model_dir)
        print("      build_index.py will automatically find them there.")
        print()


if __name__ == "__main__":
    main()


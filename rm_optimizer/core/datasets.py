"""
Dataset utilities for downloading and preparing preference datasets.

Supported datasets:
- Anthropic HH-RLHF (170K pairs)
- UltraFeedback (64K pairs)
- Stanford SHP (385K pairs)
- OpenAssistant (custom extraction)
"""

from pathlib import Path
from typing import Optional, Dict, List, Literal
import pandas as pd


DATASET_REGISTRY = {
    "hh-rlhf": {
        "hf_name": "Anthropic/hh-rlhf",
        "description": "Anthropic's Helpful and Harmless dataset",
        "size": "170K pairs",
        "columns": {"prompt": "auto", "chosen": "chosen", "rejected": "rejected"}
    },
    "ultrafeedback": {
        "hf_name": "openbmb/UltraFeedback",
        "description": "High-quality preference data with GPT-4 annotations",
        "size": "64K pairs",
        "columns": {"prompt": "instruction", "chosen": "auto", "rejected": "auto"}
    },
    "shp": {
        "hf_name": "stanfordnlp/SHP",
        "description": "Stanford Human Preferences from Reddit",
        "size": "385K pairs",
        "columns": {"prompt": "history", "chosen": "human_ref_A", "rejected": "human_ref_B"}
    },
    "helpsteer": {
        "hf_name": "nvidia/HelpSteer",
        "description": "NVIDIA's helpful steering dataset",
        "size": "37K samples",
        "columns": {"prompt": "prompt", "chosen": "response", "rejected": "auto"}
    }
}


def list_datasets() -> Dict[str, Dict]:
    """List all available datasets."""
    print("\n📦 Available Preference Datasets:\n")
    print("-" * 60)
    for name, info in DATASET_REGISTRY.items():
        print(f"  {name:15} | {info['size']:10} | {info['description']}")
    print("-" * 60)
    return DATASET_REGISTRY


def download_dataset(
    name: str = "hh-rlhf",
    output_dir: str = "data",
    split: str = "train",
    max_samples: Optional[int] = None,
    format: Literal["parquet", "json", "csv"] = "parquet"
) -> str:
    """
    Download and prepare a preference dataset.
    
    Args:
        name: Dataset name (hh-rlhf, ultrafeedback, shp, helpsteer)
        output_dir: Directory to save processed data
        split: Dataset split (train, test, validation)
        max_samples: Limit number of samples (None for all)
        format: Output format (parquet, json, csv)
    
    Returns:
        Path to saved dataset file
    
    Example:
        >>> path = download_dataset("hh-rlhf", max_samples=10000)
        >>> print(f"Dataset saved to: {path}")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    info = DATASET_REGISTRY[name]
    print(f"\n📥 Downloading {name} ({info['size']})...")
    print(f"   Source: {info['hf_name']}")
    
    # Load from HuggingFace
    dataset = load_dataset(info['hf_name'], split=split)
    
    # Limit samples if requested
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"   Sampled {max_samples} examples")
    
    # Convert to standard format
    print("   Processing...")
    df = _convert_to_standard_format(dataset, name, info)
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{name}_{split}"
    if max_samples:
        filename += f"_{max_samples // 1000}k"
    
    if format == "parquet":
        path = output_dir / f"{filename}.parquet"
        df.to_parquet(path, index=False)
    elif format == "json":
        path = output_dir / f"{filename}.json"
        df.to_json(path, orient="records", lines=True)
    else:
        path = output_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
    
    print(f"\n✅ Saved {len(df)} samples to: {path}")
    print(f"   Columns: {list(df.columns)}")
    
    return str(path)


def _convert_to_standard_format(dataset, name: str, info: Dict) -> pd.DataFrame:
    """Convert dataset to standard prompt/chosen/rejected format."""
    
    if name == "hh-rlhf":
        # HH-RLHF has conversations - extract last turn
        records = []
        for item in dataset:
            chosen = item['chosen']
            rejected = item['rejected']
            
            # Extract prompt from shared prefix
            # Format: "\n\nHuman: {prompt}\n\nAssistant: {response}"
            if "\n\nAssistant:" in chosen:
                parts = chosen.rsplit("\n\nAssistant:", 1)
                prompt = parts[0].replace("\n\nHuman:", "").strip()
                chosen_response = parts[1].strip()
            else:
                prompt = ""
                chosen_response = chosen
            
            if "\n\nAssistant:" in rejected:
                rejected_response = rejected.rsplit("\n\nAssistant:", 1)[1].strip()
            else:
                rejected_response = rejected
            
            records.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            })
        
        return pd.DataFrame(records)
    
    elif name == "ultrafeedback":
        # UltraFeedback has completions with scores
        records = []
        for item in dataset:
            prompt = item.get('instruction', item.get('prompt', ''))
            completions = item.get('completions', [])
            
            if len(completions) < 2:
                continue
            
            # Sort by score and take best/worst
            sorted_completions = sorted(
                completions, 
                key=lambda x: x.get('overall_score', 0),
                reverse=True
            )
            
            records.append({
                "prompt": prompt,
                "chosen": sorted_completions[0].get('response', ''),
                "rejected": sorted_completions[-1].get('response', '')
            })
        
        return pd.DataFrame(records)
    
    elif name == "shp":
        # SHP has A/B comparisons with labels
        records = []
        for item in dataset:
            prompt = item.get('history', '')
            
            # labels: 1 means A is better, 0 means B is better
            if item.get('labels', 1) == 1:
                chosen = item.get('human_ref_A', '')
                rejected = item.get('human_ref_B', '')
            else:
                chosen = item.get('human_ref_B', '')
                rejected = item.get('human_ref_A', '')
            
            records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        
        return pd.DataFrame(records)
    
    elif name == "helpsteer":
        # HelpSteer is single-response, create synthetic pairs
        # (use for demo, not real preference training)
        records = []
        items = list(dataset)
        
        for i in range(0, len(items) - 1, 2):
            item1, item2 = items[i], items[i + 1]
            
            # Use helpfulness score to determine preference
            score1 = item1.get('helpfulness', 0)
            score2 = item2.get('helpfulness', 0)
            
            if score1 > score2:
                chosen, rejected = item1['response'], item2['response']
            else:
                chosen, rejected = item2['response'], item1['response']
            
            records.append({
                "prompt": item1.get('prompt', ''),
                "chosen": chosen,
                "rejected": rejected
            })
        
        return pd.DataFrame(records)
    
    else:
        # Generic fallback
        df = dataset.to_pandas()
        cols = info['columns']
        
        return df.rename(columns={
            cols.get('prompt', 'prompt'): 'prompt',
            cols.get('chosen', 'chosen'): 'chosen',
            cols.get('rejected', 'rejected'): 'rejected'
        })[['prompt', 'chosen', 'rejected']]


def load_preference_data(
    path: str,
    tokenizer=None,
    max_length: int = 512
) -> pd.DataFrame:
    """
    Load preference data from file.
    
    Args:
        path: Path to data file (parquet, json, csv)
        tokenizer: Optional tokenizer for length filtering
        max_length: Maximum sequence length
    
    Returns:
        DataFrame with prompt, chosen, rejected columns
    """
    path = Path(path)
    
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in [".json", ".jsonl"]:
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    # Validate columns
    required = ['prompt', 'chosen', 'rejected']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Filter by length if tokenizer provided
    if tokenizer is not None:
        def count_tokens(text):
            return len(tokenizer.encode(str(text), add_special_tokens=False))
        
        mask = (
            df['prompt'].apply(count_tokens) + 
            df['chosen'].apply(count_tokens).clip(upper=max_length)
        ) < max_length
        
        original_len = len(df)
        df = df[mask]
        print(f"Filtered {original_len - len(df)} samples exceeding {max_length} tokens")
    
    return df


def create_demo_dataset(
    output_path: str = "data/demo.parquet",
    n_samples: int = 100
) -> str:
    """
    Create a small demo dataset for testing.
    
    Args:
        output_path: Path to save demo data
        n_samples: Number of samples
    
    Returns:
        Path to saved file
    """
    import random
    
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "How does photosynthesis work?",
        "What causes earthquakes?",
        "Describe the water cycle.",
        "What is artificial intelligence?",
        "How do vaccines work?",
        "Explain climate change.",
        "What is blockchain?",
        "How does the internet work?"
    ]
    
    good_patterns = [
        "{topic} is a complex process that involves multiple factors. Let me explain in detail...",
        "Great question! {topic} can be understood by breaking it down into key components...",
        "To understand {topic}, we need to consider the underlying principles...",
    ]
    
    bad_patterns = [
        "I don't know much about {topic}.",
        "{topic} is complicated. Google it.",
        "That's a hard question about {topic}. Maybe ask someone else.",
    ]
    
    records = []
    for i in range(n_samples):
        prompt = random.choice(prompts)
        topic = prompt.lower().replace("?", "").replace("what is ", "").replace("how does ", "")
        
        chosen = random.choice(good_patterns).format(topic=topic)
        rejected = random.choice(bad_patterns).format(topic=topic)
        
        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    df = pd.DataFrame(records)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Created demo dataset with {n_samples} samples: {output_path}")
    return output_path


# CLI integration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        list_datasets()
        print("\nUsage: python datasets.py <dataset_name> [max_samples]")
        print("Example: python datasets.py hh-rlhf 10000")
    else:
        name = sys.argv[1]
        max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None
        download_dataset(name, max_samples=max_samples)

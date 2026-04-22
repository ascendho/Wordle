from datasets import load_dataset
import pandas as pd
from predibase import Predibase

def upload_dataset(pb: Predibase, hf_repo: str, pb_name: str) -> pd.DataFrame:
    """Load dataset from HuggingFace and upload/get it on Predibase."""
    dataset = load_dataset(hf_repo, split="train")
    df = dataset.to_pandas()
    
    try:
        # Try to create a new dataset
        pb_dataset = pb.datasets.from_pandas_dataframe(df, name=pb_name)
    except Exception:
        # If it already exists, fetch it
        pb_dataset = pb.datasets.get(pb_name)
        
    return pb_dataset

def get_wordle_grpo_dataset(pb: Predibase):
    return upload_dataset(pb, "predibase/wordle-grpo", "wordle_grpo_data")

def get_wordle_sft_dataset(pb: Predibase):
    return upload_dataset(pb, "predibase/wordle-sft", "wordle_sft_data")

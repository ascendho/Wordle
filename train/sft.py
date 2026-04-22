from predibase import SFTConfig
from src.utils.config import get_predibase_client
from src.data.loader import get_wordle_sft_dataset

def run_sft_training():
    """Execute pure Supervised Fine-Tuning using the Wordle SFT dataset."""
    pb = get_predibase_client()
    dataset = get_wordle_sft_dataset(pb)
    
    pb.repos.create(name="wordle", exists_ok=True)
    
    config = SFTConfig(
        base_model="qwen2-5-7b-instruct",
        epochs=10,
        rank=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    )
    
    job = pb.finetuning.jobs.create(
        config=config,
        dataset=dataset,
        repo="wordle",
        description="Wordle SFT, 10 epochs"
    )
    print("SFT Training Job created successfully! Job ID:", getattr(job, "id", "Unknown"))

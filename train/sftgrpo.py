from predibase import (
    GRPOConfig,
    RewardFunctionsConfig,
    RewardFunctionsRuntimeConfig,
    SamplingParamsConfig,
)
from src.utils.config import get_predibase_client
from src.data.loader import get_wordle_grpo_dataset

# Import local standard reward functions from the new path
from src.rewards.format import output_format_check
from src.rewards.feedback import uses_previous_feedback
from src.rewards.entropy import guess_value

def run_sft_grpo_training(sft_version: str = "wordle/1"):
    """
    Run GRPO training initialized from a previous SFT checkpoint.
    """
    pb = get_predibase_client()
    
    # We use the GRPO training dataset
    dataset = get_wordle_grpo_dataset(pb)
    
    # Ensure the repository exists
    pb.repos.create(name="wordle", exists_ok=True)
    
    # Setup GRPO config but with different parameters (e.g. fewer generations) 
    # to continue training right after an SFT job completes
    config = GRPOConfig(
        base_model="qwen2-5-7b-instruct",
        reward_fns=RewardFunctionsConfig(
            runtime=RewardFunctionsRuntimeConfig(packages=["pandas"]),
            functions={
                "output_format_check": output_format_check,
                "uses_previous_feedback": uses_previous_feedback,
                "guess_value": guess_value,
            }
        ),
        epochs=3,
        enable_early_stopping=False,
        sampling_params=SamplingParamsConfig(max_tokens=4096),
        num_generations=8
    )
    
    # Notice we pass 'continue_from_version' with our provided specific sft version
    job = pb.finetuning.jobs.create(
        config=config,
        continue_from_version=sft_version, # Set based on SFT generation output
        dataset=dataset,
        repo="wordle",
        description="Wordle SFT+GRPO"
    )
    print(f"SFT+GRPO Training Job from {sft_version} created! Job ID: {getattr(job, 'id', 'Unknown')}")

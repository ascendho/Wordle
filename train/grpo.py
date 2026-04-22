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

def run_grpo_training():
    """Execute GRPO training initialized with reward configurations."""
    pb = get_predibase_client()
    
    # Load GRPO specific dataset
    dataset = get_wordle_grpo_dataset(pb)
    
    # Ensure the target repository exists
    pb.repos.create(name="wordle", exists_ok=True)
    
    # Set up GRPO training configuration parameters
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
        sampling_params=SamplingParamsConfig(max_tokens=4096),
        num_generations=16
    )
    
    # Start pure GRPO training
    job = pb.finetuning.jobs.create(
        config=config,
        dataset=dataset,
        repo="wordle",
        description="Wordle GRPO"
    )
    print("GRPO Training Job created successfully! Job ID:", getattr(job, "id", "Unknown"))

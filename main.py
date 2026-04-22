import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Wordle SFT/GRPO Fine-Tuning Pipeline CLI")
    
    parser.add_argument(
        "--run", 
        type=str, 
        choices=["data", "train", "eval"], 
        required=True,
        help="The pipeline stage to run: target data prep, training execution, or evaluation benchmark."
    )
    
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["sft", "grpo", "sftgrpo"], 
        default="grpo",
        help="The specific type of training execution (only applies if --run train)."
    )
    
    parser.add_argument(
        "--adapter", 
        type=str, 
        default="wordle/1",
        help="Adapter / Model version ID to evaluate or use as base (e.g., wordle/1)"
    )

    args = parser.parse_args()

    if args.run == "data":
        print("Preparing datasets for Wordle...")
        # Local Imports
        from utils.config import get_predibase_client
        from data.loader import get_wordle_grpo_dataset, get_wordle_sft_dataset
        
        pb = get_predibase_client()
        get_wordle_sft_dataset(pb)
        get_wordle_grpo_dataset(pb)
        print("Datasets retrieved and fetched successfully.")

    elif args.run == "train":
        if args.type == "sft":
            from train.sft import run_sft_training
            run_sft_training()
            
        elif args.type == "grpo":
            from train.grpo import run_grpo_training
            run_grpo_training()
            
        elif args.type == "sftgrpo":
            from train.sftgrpo import run_sft_grpo_training
            run_sft_grpo_training(sft_version=args.adapter)

    elif args.run == "eval":
        from eval.evaluate import run_evaluation
        run_evaluation(adapter_id=args.adapter)

if __name__ == "__main__":
    main()

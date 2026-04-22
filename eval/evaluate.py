import pandas as pd
import re
import ast

from src.utils.config import get_predibase_client

def extract_guess(completion: str) -> str:
    """Helper to extract the string inside <guess> tags."""
    match = re.search(r"<guess>\s*([\s\S]*?)\s*<\/guess>", completion, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_secret_word(example_row: pd.Series) -> str:
    """Try common field names used for target word labels in Wordle datasets."""
    for key in ("secret", "target", "answer", "word", "secret_word"):
        value = example_row.get(key)
        if isinstance(value, str) and len(value.strip()) == 5:
            return value.strip().upper()
    return ""


def safe_history_len(raw_history) -> int:
    """Parse past_guess_history safely and return the number of previous guesses."""
    if isinstance(raw_history, list):
        return len(raw_history)
    if not isinstance(raw_history, str) or not raw_history.strip():
        return 0
    try:
        parsed = ast.literal_eval(raw_history)
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0

def run_evaluation(adapter_id: str):
    """
    Evaluates the model by submitting prompts from a test subset,
    generating responses via the Predibase prediction API, and tabulating results.
    """
    print(f"--- Benchmarking / Evaluating Model: {adapter_id} ---")
    pb = get_predibase_client()
    
    try:
        # 1. Fetch our base test dataset or examples
        # Here we just fetch the top 10 items for a quick eval loop.
        dataset = pb.datasets.get("wordle_grpo_data")
        df = dataset.to_pandas().head(10)
        
        # 2. Get the deployed model or adapter we want to test
        # In a real environment you must deploy the adapter first (pb.deployments...).
        # We will assume a mock/lorx text generation endpoint:
        try:
            adapter = pb.adapters.get(adapter_id)
        except Exception as e:
            print(f"Warning: Adapter {adapter_id} not available or deployed. {e}")
            print("Note: To run real eval, inference must be setup. Generating simulated loop structure...")
            return

        total_games = len(df)
        solved_count = 0
        guesses_in_solved = []

        print(f"Executing {total_games} Wordle prediction tests...")
        for index, row in df.iterrows():
            prompt = row['prompt']
            expected_secret = extract_secret_word(row)
            if not expected_secret:
                # Skip rows without an explicit target word; they cannot be scored for solved rate.
                continue
            
            # 3. Call the inference endpoint
            response = adapter.generate(
                prompt=prompt,
                max_new_tokens=512,
            )
            
            # Extract content text (depending on generation response structure)
            completion = response.generated_text if hasattr(response, 'generated_text') else str(response)
            
            guess = extract_guess(completion)
            
            # Simple check logic for demonstration
            if guess.upper() == expected_secret:
                solved_count += 1
                guesses_in_solved.append(safe_history_len(row.get('past_guess_history')) + 1)
        
        avg_guesses = sum(guesses_in_solved) / len(guesses_in_solved) if guesses_in_solved else 0.0

        results = {
            "Model": adapter_id,
            "Solved Games": solved_count,
            "Total Evaluation Run": total_games,
            "Avg # Guesses (In solved games)": round(avg_guesses, 2)
        }
        
        results_df = pd.DataFrame([results])
        print("\n--- Evaluation Benchmark Results ---")
        print(results_df.to_string(index=False))
        print("------------------------------------\n")
    except Exception as e:
        print(f"Evaluation error: {e}")

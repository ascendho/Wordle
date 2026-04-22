# Wordle Model Fine-tuning Pipeline

This Python repository modularizes the Wordle Supervised Fine-Tuning (SFT) and Generative Reward Policy Optimization (GRPO) implementation built on Predibase and Qwen 2.5 7B Instruct. It replicates the course configuration under standard Python `src-layout` project bounds.

## Requirements
1. Copy `.env.example` -> `.env` and fill your `PREDIBASE_API_KEY`
2. Run `pip install -r requirements.txt`

## Execution via `main.py`
The project relies on `main.py` acting as an endpoint for all operations.

**1. Data Loading**
Cache HuggingFace Datasets onto Predibase:
```bash
python main.py --run data
```

**2. Distributed Training Jobs**
Trigger models training independently:
```bash
python main.py --run train --type sft
python main.py --run train --type grpo

# Continue GRPO training from an SFT model version output
python main.py --run train --type sftgrpo --adapter wordle/1
```

**3. Benchmark & Evaluate**
Simulates evaluation calculation to extract Solved Rates & Guesses constraints:
```bash
python main.py --run eval --adapter wordle/1
```

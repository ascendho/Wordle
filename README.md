# Wordle Model Fine-tuning Pipeline

This Python repository modularizes the Wordle Supervised Fine-Tuning (SFT) and Generative Reward Policy Optimization (GRPO) implementation built on Predibase and Qwen 2.5 7B Instruct.

The project uses a hybrid layout:
- Root level keeps pipeline actions (`main.py`, `train/`, `eval/`) easy to access.
- `src/` contains reusable shared modules (`data/`, `rewards/`, `utils/`).

## Requirements
1. Copy `.env.example` -> `.env` and fill your `PREDIBASE_API_KEY`
2. Run `pip install -r requirements.txt`

## Execution via `main.py`
The project relies on `main.py` acting as an endpoint for all operations.

## Project Structure
```text
.
├── main.py
├── train/
│   ├── sft.py
│   ├── grpo.py
│   └── sftgrpo.py
├── eval/
│   └── evaluate.py
└── src/
	├── data/
	│   └── loader.py
	├── rewards/
	│   ├── format.py
	│   ├── feedback.py
	│   └── entropy.py
	└── utils/
		└── config.py
```

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
Run evaluation logic to compute solved games and average guesses from model outputs:
```bash
python main.py --run eval --adapter wordle/1
```

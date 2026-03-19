# Safety Drift Study

A checkpoint-level safety regression framework for fine-tuned LLMs.

## Research Question
How does fine-tuning an instruction-tuned LLM on domain-specific data
unintentionally weaken safety behavior mid-training — and why does
snapshot evaluation (base vs final model) fail to capture this?

## Models
| Key | Model | Size | Family |
|---|---|---|---|
| qwen0.5b | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | Qwen |
| qwen1.5b | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Qwen |
| smollm1.7b | HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B | SmolLM |

## Datasets
| Key | Dataset | Domain |
|---|---|---|
| dialogsum | knkarthick/dialogsum | Conversation |
| xsum | EdinburghNLP/xsum | News |

## Safety Benchmarks
- **Refusal Rate**: LibrAI/do-not-answer (939 prompts)
- **Jailbreak ASR**: lmsys/toxic-chat (113 real jailbreak attempts)
- **Toxicity**: Detoxify scoring on all responses

## Metrics
1. **Refusal Rate** — % of harmful prompts the model refuses
2. **Jailbreak ASR** — % of jailbreak attempts that succeed
3. **Avg Toxicity** — Mean Detoxify toxicity score across responses

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Single Experiment
```bash
python3 main.py --model qwen1.5b --dataset dialogsum
```

## Run All Experiments
```bash
bash scripts/run_experiments.sh
```

## Run Smoke Test
```bash
python3 scripts/smoke_test.py
```

## Compare Results
```bash
python3 scripts/compare_results.py
```

## Project Structure
```
safety-drift-study/
├── config.py                  # All hyperparameters
├── main.py                    # Main pipeline
├── evaluation/
│   └── __init__.py            # Safety evaluation pipeline
├── scripts/
│   ├── smoke_test.py          # Quick pipeline verification
│   ├── run_experiments.sh     # Run all 6 experiments
│   ├── compare_results.py     # Cross-experiment comparison
│   └── test_plots.py          # Preview drift curves
├── checkpoints/               # Saved model checkpoints
├── results/                   # JSON results + plots
├── logs/                      # Training logs
└── requirements.txt
```

## Results Structure
```
results/
├── qwen0.5b_dialogsum/
│   ├── safety_drift_results.json
│   ├── safety_drift_curves.png
│   └── release_gate_report.json
├── qwen1.5b_dialogsum/
├── smollm1.7b_dialogsum/
└── comparison/
    ├── comparison_all.png
    └── qwen0.5b_comparison.png
```
# scripts/smoke_test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from evaluation import evaluate_checkpoint, check_release_gate
import evaluation
from config import (
    MODELS, DATASETS, BASE_DIR, BATCH_SIZE,
    GRAD_ACCUM_STEPS, LEARNING_RATE, MAX_SEQ_LENGTH,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, DEVICE, DATALOADER_PIN_MEMORY,
)

# ── Smoke test overrides ──
SMOKE_MODEL_KEY      = "qwen1.5b"
SMOKE_DATASET_KEY    = "dialogsum"
SMOKE_TRAIN_SAMPLES  = 100
SMOKE_EPOCHS         = 1
SMOKE_SAVE_STEPS     = 10
SMOKE_SAVE_LIMIT     = 3
SMOKE_HARMFUL_EVAL   = 5
SMOKE_JAILBREAK_EVAL = 4
SMOKE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_smoke")
SMOKE_RESULTS_DIR    = os.path.join(BASE_DIR, "results", "smoke")


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def run_smoke_test():
    os.makedirs(SMOKE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SMOKE_RESULTS_DIR, exist_ok=True)

    print("\n" + "="*50)
    print("🔬 SMOKE TEST — Tiny run to verify pipeline")
    print("="*50)
    print(f"   Model         : {SMOKE_MODEL_KEY}")
    print(f"   Dataset       : {SMOKE_DATASET_KEY}")
    print(f"   Train samples : {SMOKE_TRAIN_SAMPLES}")
    print(f"   Epochs        : {SMOKE_EPOCHS}")
    print(f"   Save every    : {SMOKE_SAVE_STEPS} steps")
    print(f"   Device        : {DEVICE}\n")

    # ── Step 1: Dataset ──
    print("📦 Step 1: Loading dataset...")
    ds_config = DATASETS[SMOKE_DATASET_KEY]
    text_col  = ds_config["text_col"]
    sum_col   = ds_config["summary_col"]
    prompt    = ds_config["prompt"]

    dataset = load_dataset(ds_config["path"], split="train")
    dataset = dataset.filter(lambda x: len(x[sum_col]) > 10)

    def format_sample(example):
        return {
            "text": (
                f"<|im_start|>user\n"
                f"{prompt} {example[text_col][:800]}"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{example[sum_col]}"
                f"<|im_end|>"
            )
        }

    dataset = dataset.map(format_sample)
    dataset = dataset.filter(lambda x: len(x["text"]) < 1200)
    dataset = dataset.select(range(SMOKE_TRAIN_SAMPLES))
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Train: {len(split['train'])} | Val: {len(split['test'])}")

    # ── Step 2: Model ──
    print("\n🤖 Step 2: Loading model...")
    model_name = MODELS[SMOKE_MODEL_KEY]["path"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=DEVICE
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"✅ Model loaded on {DEVICE}")

    # ── Step 3: Fine-tune ──
    print("\n🚀 Step 3: Fine-tuning (smoke)...")

    def formatting_func(example):
        return example["text"]

    training_args = TrainingArguments(
        output_dir=SMOKE_CHECKPOINT_DIR,
        num_train_epochs=SMOKE_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_strategy="steps",
        save_steps=SMOKE_SAVE_STEPS,
        save_total_limit=SMOKE_SAVE_LIMIT,
        eval_strategy="steps",
        eval_steps=SMOKE_SAVE_STEPS,
        load_best_model_at_end=False,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"].select_columns(["text"]),
        eval_dataset=split["test"].select_columns(["text"]),
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    trainer.train()
    print("✅ Fine-tuning complete!")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Step 4: Evaluate ──
    print("\n📊 Step 4: Evaluating checkpoints...")

    # Override eval sizes for speed
    original_harmful   = evaluation.HARMFUL_PROMPTS
    original_jailbreak = evaluation.JAILBREAK_PROMPTS
    evaluation.HARMFUL_PROMPTS   = evaluation.HARMFUL_PROMPTS[:SMOKE_HARMFUL_EVAL]
    evaluation.JAILBREAK_PROMPTS = evaluation.JAILBREAK_PROMPTS[:SMOKE_JAILBREAK_EVAL]

    all_results = []

    # Base model
    print("\n  Evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=DEVICE
    )
    result = evaluate_checkpoint(base_model, tokenizer, step=0)
    all_results.append(result)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Checkpoints
    checkpoints = sorted(
        [d for d in os.listdir(SMOKE_CHECKPOINT_DIR)
         if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    print(f"\n  Found {len(checkpoints)} smoke checkpoints")

    for ckpt in checkpoints:
        step      = int(ckpt.split("-")[1])
        ckpt_path = os.path.join(SMOKE_CHECKPOINT_DIR, ckpt)
        ckpt_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE
        )
        ckpt_model = PeftModel.from_pretrained(ckpt_model, ckpt_path)
        result     = evaluate_checkpoint(ckpt_model, tokenizer, step=step)
        all_results.append(result)
        del ckpt_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Restore
    evaluation.HARMFUL_PROMPTS   = original_harmful
    evaluation.JAILBREAK_PROMPTS = original_jailbreak

    # ── Step 5: Results ──
    print("\n" + "="*50)
    print("📋 SMOKE TEST RESULTS")
    print("="*50)
    print(f"{'Step':>6} | {'Refusal':>8} | {'ASR':>8} | {'Toxicity':>10}")
    print("-" * 42)
    for r in all_results:
        print(f"{r['step']:>6} | {r['refusal_rate']:>8.3f} | "
              f"{r['jailbreak_asr']:>8.3f} | {r['avg_toxicity']:>10.4f}")

    print("\n🚦 Release Gate:")
    flagged = check_release_gate(all_results)

    results_path = os.path.join(SMOKE_RESULTS_DIR, "smoke_test_results.json")
    with open(results_path, "w") as f:
        json.dump(
            make_serializable(
                [{k: v for k, v in r.items() if k != "details"}
                 for r in all_results]
            ),
            f, indent=2
        )

    print("\n" + "="*50)
    print("✅ SMOKE TEST COMPLETE")
    print("="*50)
    print(f"   Checkpoints evaluated : {len(all_results)}")
    print(f"   Flagged               : {len(flagged)}")
    print(f"   Results saved to      : {results_path}")
    print("\n🚀 Pipeline is ready for full run via main.py")


if __name__ == "__main__":
    run_smoke_test()
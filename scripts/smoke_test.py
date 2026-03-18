# scripts/smoke_test.py
# Runs the full pipeline on tiny data to verify everything works
# before the full overnight run

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from evaluation import evaluate_checkpoint, check_release_gate
from config import *


# ── Smoke test overrides (tiny values) ──
SMOKE_TRAIN_SAMPLES  = 100    # instead of 5000
SMOKE_EPOCHS         = 1      # instead of 3
SMOKE_SAVE_STEPS     = 10     # instead of 100
SMOKE_SAVE_LIMIT     = 3      # only keep 3 checkpoints
SMOKE_HARMFUL_EVAL   = 5      # instead of 50
SMOKE_JAILBREAK_EVAL = 4      # instead of 30
SMOKE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_smoke")


def run_smoke_test():
    print("\n" + "="*50)
    print("🔬 SMOKE TEST — Tiny run to verify pipeline")
    print("="*50)
    print(f"   Train samples : {SMOKE_TRAIN_SAMPLES}")
    print(f"   Epochs        : {SMOKE_EPOCHS}")
    print(f"   Save every    : {SMOKE_SAVE_STEPS} steps")
    print(f"   Device        : {DEVICE}\n")

    os.makedirs(SMOKE_CHECKPOINT_DIR, exist_ok=True)

    # ─────────────────────────────────────────
    # 1. Load tiny dataset
    # ─────────────────────────────────────────
    print("📦 Step 1: Loading dataset...")
    dataset = load_dataset("knkarthick/dialogsum", split="train")

    def format_sample(example):
        return {
            "text": (
                f"<|im_start|>user\n"
                f"Summarize this conversation: {example['dialogue'][:800]}"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{example['summary']}"
                f"<|im_end|>"
            )
        }

    dataset = dataset.filter(lambda x: len(x["summary"]) > 10)
    dataset = dataset.map(format_sample)
    dataset = dataset.filter(lambda x: len(x["text"]) < 1200)
    dataset = dataset.select(range(SMOKE_TRAIN_SAMPLES))
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Train: {len(split['train'])} | Val: {len(split['test'])}")

    # ─────────────────────────────────────────
    # 2. Load model
    # ─────────────────────────────────────────
    print("\n🤖 Step 2: Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"✅ Model loaded on {DEVICE}")

    # ─────────────────────────────────────────
    # 3. Fine-tune (tiny)
    # ─────────────────────────────────────────
    print("\n🚀 Step 3: Fine-tuning (smoke)...")

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
    )

    def formatting_func(example):
        return example["text"]

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

    # Free memory
    del model
    torch.mps.empty_cache()

    # ─────────────────────────────────────────
    # 4. Evaluate checkpoints
    # ─────────────────────────────────────────
    print("\n📊 Step 4: Evaluating checkpoints...")

    # Temporarily override eval counts
    import evaluation
    original_harmful  = evaluation.HARMFUL_PROMPTS
    original_jailbreak = evaluation.JAILBREAK_PROMPTS

    # Use tiny subsets for smoke test
    evaluation.HARMFUL_PROMPTS    = evaluation.HARMFUL_PROMPTS[:SMOKE_HARMFUL_EVAL]
    evaluation.JAILBREAK_PROMPTS = evaluation.JAILBREAK_PROMPTS[:2]
    all_results = []

    # Evaluate base model
    print("\n  Evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    result = evaluate_checkpoint(base_model, tokenizer, step=0)
    all_results.append(result)
    del base_model
    torch.mps.empty_cache()

    # Evaluate each checkpoint
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
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE
        )
        ckpt_model = PeftModel.from_pretrained(ckpt_model, ckpt_path)

        result = evaluate_checkpoint(ckpt_model, tokenizer, step=step)
        all_results.append(result)

        del ckpt_model
        torch.mps.empty_cache()

    # Restore original prompt lists
    evaluation.HARMFUL_PROMPTS     = original_harmful
    evaluation.JAILBREAK_PROMPTS = original_jailbreak

    # ─────────────────────────────────────────
    # 5. Save + print results
    # ─────────────────────────────────────────
    print("\n" + "="*50)
    print("📋 SMOKE TEST RESULTS")
    print("="*50)
    print(f"{'Step':>6} | {'Refusal':>8} | {'ASR':>8} | {'Toxicity':>10}")
    print("-" * 42)
    for r in all_results:
        print(f"{r['step']:>6} | {r['refusal_rate']:>8.3f} | "
              f"{r['jailbreak_asr']:>8.3f} | {r['avg_toxicity']:>10.4f}")

    # Apply release gate
    print("\n🚦 Release Gate:")
    flagged = check_release_gate(all_results)

    # Save smoke results
    smoke_results_path = os.path.join(RESULTS_DIR, "smoke_test_results.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def make_serializable(obj):
        """Convert numpy/torch types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # catches numpy float32, int64 etc.
            return obj.item()
        else:
            return obj

    with open(smoke_results_path, "w") as f:
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
    print(f"   Results saved to      : {smoke_results_path}")
    print("\n🚀 Pipeline is ready for full run via main.py")


if __name__ == "__main__":
    run_smoke_test()
# main.py
import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from datasets import load_dataset
from config import *
from evaluation import evaluate_checkpoint, check_release_gate


# ─────────────────────────────────────────────
# STEP 1: Load and prepare dataset
# ─────────────────────────────────────────────
def prepare_dataset():
    print("\n" + "="*50)
    print("STEP 1: Loading Dataset")
    print("="*50)

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
    dataset = dataset.select(range(NUM_TRAIN_SAMPLES))
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"✅ Train samples : {len(split['train'])}")
    print(f"✅ Val samples   : {len(split['test'])}")
    return split


# ─────────────────────────────────────────────
# STEP 2: Load base model + apply LoRA
# ─────────────────────────────────────────────
def load_model_and_tokenizer():
    print("\n" + "="*50)
    print("STEP 2: Loading Model")
    print("="*50)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

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

    print(f"✅ Model loaded  : {MODEL_NAME}")
    print(f"✅ Device        : {DEVICE}")
    return model, tokenizer


# ─────────────────────────────────────────────
# STEP 3: Fine-tune with checkpoint saving
# ─────────────────────────────────────────────
def fine_tune(model, tokenizer, dataset_split):
    print("\n" + "="*50)
    print("STEP 3: Fine-Tuning")
    print("="*50)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        peft_config=None,
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning complete!")


# ─────────────────────────────────────────────
# STEP 4: Evaluate all checkpoints
# ─────────────────────────────────────────────
def evaluate_all_checkpoints(tokenizer):
    print("\n" + "="*50)
    print("STEP 4: Evaluating All Checkpoints")
    print("="*50)

    all_results = []

    # -- Evaluate base model first (step 0) --
    print("\n📊 Evaluating base model (step 0)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    result = evaluate_checkpoint(base_model, tokenizer, step=0)
    all_results.append(result)
    del base_model
    torch.mps.empty_cache()

    # -- Evaluate each saved checkpoint --
    checkpoints = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )

    print(f"\n📂 Found {len(checkpoints)} checkpoints to evaluate")

    for ckpt in checkpoints:
        step = int(ckpt.split("-")[1])
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)

        # Load base + LoRA adapter for this checkpoint
        model_ckpt = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE
        )
        model_ckpt = PeftModel.from_pretrained(model_ckpt, ckpt_path)

        result = evaluate_checkpoint(model_ckpt, tokenizer, step=step)
        all_results.append(result)

        # Critical: free memory after each checkpoint eval
        del model_ckpt
        torch.mps.empty_cache()

    # -- Save results to JSON --
    results_path = os.path.join(RESULTS_DIR, "safety_drift_results.json")
    # Remove details before saving to keep file small
    results_to_save = [
        {k: v for k, v in r.items() if k != "details"}
        for r in all_results
    ]
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n✅ Results saved to {results_path}")

    return all_results


# ─────────────────────────────────────────────
# STEP 5: Plot safety drift curves
# ─────────────────────────────────────────────
def plot_drift_curves(all_results):
    print("\n" + "="*50)
    print("STEP 5: Plotting Safety Drift Curves")
    print("="*50)

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "details"}
        for r in all_results
    ])
    df = df.sort_values("step").reset_index(drop=True)

    base_refusal  = df.iloc[0]["refusal_rate"]
    base_asr      = df.iloc[0]["jailbreak_asr"]
    base_toxicity = df.iloc[0]["avg_toxicity"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Safety Drift Across Fine-Tuning Checkpoints", fontsize=14)

    # Plot 1: Refusal Rate
    axes[0].plot(df["step"], df["refusal_rate"],
                 marker="o", color="steelblue", label="Refusal Rate")
    axes[0].axhline(base_refusal, linestyle="--",
                    color="red", alpha=0.5, label="Base model")
    axes[0].set_ylabel("Refusal Rate ↑ (higher = safer)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Jailbreak ASR
    axes[1].plot(df["step"], df["jailbreak_asr"],
                 marker="s", color="darkorange", label="Jailbreak ASR")
    axes[1].axhline(base_asr, linestyle="--",
                    color="red", alpha=0.5, label="Base model")
    axes[1].set_ylabel("Jailbreak ASR ↓ (lower = safer)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Toxicity
    axes[2].plot(df["step"], df["avg_toxicity"],
                 marker="^", color="purple", label="Avg Toxicity")
    axes[2].axhline(base_toxicity, linestyle="--",
                    color="red", alpha=0.5, label="Base model")
    axes[2].set_ylabel("Avg Toxicity ↓ (lower = safer)")
    axes[2].set_xlabel("Training Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "safety_drift_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved to {plot_path}")


# ─────────────────────────────────────────────
# STEP 6: Apply release gate
# ─────────────────────────────────────────────
def apply_release_gate(all_results):
    print("\n" + "="*50)
    print("STEP 6: Applying Release Gate")
    print("="*50)

    flagged = check_release_gate(all_results)

    gate_path = os.path.join(RESULTS_DIR, "release_gate_report.json")
    with open(gate_path, "w") as f:
        json.dump(flagged, f, indent=2)

    print(f"\n✅ Gate report saved to {gate_path}")
    print(f"   Flagged checkpoints : {len(flagged)}")
    print(f"   Safe checkpoints    : {len(all_results) - 1 - len(flagged)}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Safety Drift Study — Starting Pipeline")
    print(f"   Model  : {MODEL_NAME}")
    print(f"   Device : {DEVICE}")
    print(f"   Epochs : {NUM_EPOCHS}")
    print(f"   Steps  : save every {SAVE_STEPS} steps\n")

    # Run all steps in order
    dataset_split        = prepare_dataset()
    model, tokenizer     = load_model_and_tokenizer()
    fine_tune(model, tokenizer, dataset_split)
    all_results          = evaluate_all_checkpoints(tokenizer)
    plot_drift_curves(all_results)
    apply_release_gate(all_results)

    print("\n✅ Pipeline complete!")
    print(f"   Results → {RESULTS_DIR}")
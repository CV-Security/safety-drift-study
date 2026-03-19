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
from config import (
    get_config, MODELS, DATASETS, get_experiment_dirs,
    NUM_TRAIN_SAMPLES, MAX_SEQ_LENGTH, BATCH_SIZE,
    GRAD_ACCUM_STEPS, LEARNING_RATE, NUM_EPOCHS,
    SAVE_STEPS, SAVE_TOTAL_LIMIT, LORA_R, LORA_ALPHA,
    LORA_DROPOUT, LORA_TARGET_MODULES, DEVICE,
    DATALOADER_PIN_MEMORY
)
from evaluation import evaluate_checkpoint, check_release_gate


def make_serializable(obj):
    """Convert numpy/torch types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
def prepare_dataset(dataset_key):
    print("\n" + "="*50)
    print(f"STEP 1: Loading Dataset — {dataset_key}")
    print("="*50)

    ds_config = DATASETS[dataset_key]
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

    if len(dataset) > NUM_TRAIN_SAMPLES:
        dataset = dataset.select(range(NUM_TRAIN_SAMPLES))

    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Train samples : {len(split['train'])}")
    print(f"✅ Val samples   : {len(split['test'])}")
    return split


# ─────────────────────────────────────────────
# STEP 2: Load Model + LoRA
# ─────────────────────────────────────────────
def load_model_and_tokenizer(model_key):
    print("\n" + "="*50)
    print(f"STEP 2: Loading Model — {model_key}")
    print("="*50)

    model_config = MODELS[model_key]
    model_name   = model_config["path"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
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

    print(f"✅ Model   : {model_name}")
    print(f"✅ Family  : {model_config['family']}")
    print(f"✅ Size    : {model_config['size']}")
    print(f"✅ Device  : {DEVICE}")
    return model, tokenizer, model_name


# ─────────────────────────────────────────────
# STEP 3: Fine-Tune
# ─────────────────────────────────────────────
def fine_tune(model, tokenizer, dataset_split, checkpoint_dir):
    print("\n" + "="*50)
    print("STEP 3: Fine-Tuning")
    print("="*50)

    def formatting_func(example):
        return example["text"]

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
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
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"].select_columns(["text"]),
        eval_dataset=dataset_split["test"].select_columns(["text"]),
        processing_class=tokenizer,
        formatting_func=formatting_func,
        peft_config=None,
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning complete!")


# ─────────────────────────────────────────────
# STEP 4: Evaluate All Checkpoints
# ─────────────────────────────────────────────
def evaluate_all_checkpoints(tokenizer, model_name, checkpoint_dir, results_dir):
    print("\n" + "="*50)
    print("STEP 4: Evaluating All Checkpoints")
    print("="*50)

    all_results = []

    # Evaluate base model first
    print("\n📊 Evaluating base model (step 0)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=DEVICE
    )
    result = evaluate_checkpoint(base_model, tokenizer, step=0)
    all_results.append(result)
    del base_model
    torch.mps.empty_cache()

    # Evaluate each saved checkpoint
    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir)
         if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    print(f"\n📂 Found {len(checkpoints)} checkpoints")

    for ckpt in checkpoints:
        step      = int(ckpt.split("-")[1])
        ckpt_path = os.path.join(checkpoint_dir, ckpt)

        ckpt_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=DEVICE
        )
        ckpt_model = PeftModel.from_pretrained(ckpt_model, ckpt_path)
        result     = evaluate_checkpoint(ckpt_model, tokenizer, step=step)
        all_results.append(result)

        del ckpt_model
        torch.mps.empty_cache()

    # Save results
    results_path = os.path.join(results_dir, "safety_drift_results.json")
    with open(results_path, "w") as f:
        json.dump(
            make_serializable(
                [{k: v for k, v in r.items() if k != "details"}
                 for r in all_results]
            ),
            f, indent=2
        )
    print(f"\n✅ Results saved → {results_path}")
    return all_results


# ─────────────────────────────────────────────
# STEP 5: Plot Drift Curves
# ─────────────────────────────────────────────
def plot_drift_curves(all_results, results_dir, experiment_name):
    print("\n" + "="*50)
    print("STEP 5: Plotting Safety Drift Curves")
    print("="*50)

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "details"}
        for r in all_results
    ]).sort_values("step").reset_index(drop=True)

    base_refusal  = df.iloc[0]["refusal_rate"]
    base_asr      = df.iloc[0]["jailbreak_asr"]
    base_toxicity = df.iloc[0]["avg_toxicity"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Safety Drift — {experiment_name}",
        fontsize=14, fontweight="bold"
    )

    # Refusal Rate
    axes[0].plot(df["step"], df["refusal_rate"],
                 marker="o", color="steelblue",
                 linewidth=2, markersize=7, label="Refusal Rate")
    axes[0].axhline(base_refusal, linestyle="--", color="red",
                    alpha=0.6, label=f"Base ({base_refusal:.2f})")
    axes[0].axhspan(base_refusal - 0.10, base_refusal,
                    alpha=0.1, color="orange", label="Warning zone")
    axes[0].set_ylabel("Refusal Rate ↑", fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Jailbreak ASR
    axes[1].plot(df["step"], df["jailbreak_asr"],
                 marker="s", color="darkorange",
                 linewidth=2, markersize=7, label="Jailbreak ASR")
    axes[1].axhline(base_asr, linestyle="--", color="red",
                    alpha=0.6, label=f"Base ({base_asr:.2f})")
    axes[1].axhspan(base_asr, base_asr + 0.15,
                    alpha=0.1, color="orange", label="Warning zone")
    axes[1].set_ylabel("Jailbreak ASR ↓", fontsize=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Toxicity
    axes[2].plot(df["step"], df["avg_toxicity"],
                 marker="^", color="purple",
                 linewidth=2, markersize=7, label="Avg Toxicity")
    axes[2].axhline(base_toxicity, linestyle="--", color="red",
                    alpha=0.6, label=f"Base ({base_toxicity:.2f})")
    axes[2].axhspan(base_toxicity, base_toxicity + 0.05,
                    alpha=0.1, color="orange", label="Warning zone")
    axes[2].set_ylabel("Avg Toxicity ↓", fontsize=10)
    axes[2].set_xlabel("Training Step", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "safety_drift_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved → {plot_path}")


# ─────────────────────────────────────────────
# STEP 6: Release Gate
# ─────────────────────────────────────────────
def apply_release_gate(all_results, results_dir):
    print("\n" + "="*50)
    print("STEP 6: Applying Release Gate")
    print("="*50)

    flagged   = check_release_gate(all_results)
    gate_path = os.path.join(results_dir, "release_gate_report.json")

    with open(gate_path, "w") as f:
        json.dump(make_serializable(flagged), f, indent=2)

    print(f"\n✅ Gate report saved → {gate_path}")
    print(f"   Flagged  : {len(flagged)}")
    print(f"   Safe     : {len(all_results) - 1 - len(flagged)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args        = get_config()
    model_key   = args.model
    dataset_key = args.dataset
    experiment  = f"{model_key}_{dataset_key}"

    checkpoint_dir, results_dir, logs_dir = get_experiment_dirs(
        model_key, dataset_key
    )

    print("\n🔬 Safety Drift Study")
    print(f"   Experiment : {experiment}")
    print(f"   Model      : {MODELS[model_key]['path']}")
    print(f"   Dataset    : {DATASETS[dataset_key]['path']}")
    print(f"   Device     : {DEVICE}")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   Results    : {results_dir}")

    dataset_split                = prepare_dataset(dataset_key)
    model, tokenizer, model_name = load_model_and_tokenizer(model_key)
    fine_tune(model, tokenizer, dataset_split, checkpoint_dir)

    del model
    torch.mps.empty_cache()

    all_results = evaluate_all_checkpoints(
        tokenizer, model_name, checkpoint_dir, results_dir
    )
    plot_drift_curves(all_results, results_dir, experiment)
    apply_release_gate(all_results, results_dir)

    print(f"\n✅ Experiment complete: {experiment}")
    print(f"   Results → {results_dir}")
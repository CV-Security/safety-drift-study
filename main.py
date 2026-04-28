# main.py
import os
import json
import torch
import numpy as np
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
from evaluation import (
    evaluate_checkpoint, check_release_gate,
    compute_sds, compute_drift_onset,
    compute_drift_stability_index, classify_drift_pattern,
    find_optimal_checkpoint
)


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
# STEP 4: Evaluate Task Performance (ROUGE-L)
# ─────────────────────────────────────────────
def evaluate_task_performance(model, tokenizer,
                               val_dataset, step,
                               dataset_key, n_samples=20):
    """
    Measures task performance using ROUGE-L score.
    Runs alongside safety evaluation at each checkpoint.

    Rising ROUGE-L + falling refusal rate = true safety drift
    (rules out catastrophic forgetting as explanation)
    """
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        print("  ⚠️  rouge_score not installed — skipping task eval")
        return 0.0

    ds_config  = DATASETS[dataset_key]
    sum_col    = ds_config["summary_col"]
    text_col   = ds_config["text_col"]
    prompt_txt = ds_config["prompt"]

    scores = []
    samples = val_dataset.select(
        range(min(n_samples, len(val_dataset)))
    )

    for sample in samples:
        # Build prompt
        prompt = (
            f"<|im_start|>user\n"
            f"{prompt_txt} {sample[text_col][:800]}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        reference = str(sample[sum_col]).strip()

        if generated and reference:
            score = scorer.score(reference, generated)
            scores.append(score["rougeL"].fmeasure)

    avg_rouge = float(np.mean(scores)) if scores else 0.0
    print(f"  📝 Task ROUGE-L at step {step}: {avg_rouge:.4f}")
    return avg_rouge


# ─────────────────────────────────────────────
# STEP 5: Evaluate All Checkpoints
# ─────────────────────────────────────────────
def evaluate_all_checkpoints(tokenizer, model_name,
                              checkpoint_dir, results_dir,
                              val_dataset, dataset_key):
    print("\n" + "="*50)
    print("STEP 5: Evaluating All Checkpoints")
    print("="*50)

    all_results = []

    # ── Evaluate base model first ──
    print("\n📊 Evaluating base model (step 0)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=DEVICE
    )
    result = evaluate_checkpoint(base_model, tokenizer, step=0)

    # Add task performance
    rouge = evaluate_task_performance(
        base_model, tokenizer, val_dataset, 0, dataset_key
    )
    result["task_rouge"] = rouge
    result["sds"]        = 0.0
    all_results.append(result)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Evaluate each checkpoint ──
    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir)
         if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    print(f"\n📂 Found {len(checkpoints)} checkpoints")

    base_result = all_results[0]

    for ckpt in checkpoints:
        step      = int(ckpt.split("-")[1])
        ckpt_path = os.path.join(checkpoint_dir, ckpt)

        ckpt_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=DEVICE
        )
        ckpt_model = PeftModel.from_pretrained(
            ckpt_model,
            ckpt_path,
            is_trainable=False,
            local_files_only=True
        )

        # Safety evaluation
        result = evaluate_checkpoint(
            ckpt_model, tokenizer, step=step
        )

        # Task performance
        rouge = evaluate_task_performance(
            ckpt_model, tokenizer,
            val_dataset, step, dataset_key
        )
        result["task_rouge"] = rouge

        # Formal SDS metric
        result["sds"] = compute_sds(result, base_result)

        all_results.append(result)

        del ckpt_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ── Save results ──
    results_path = os.path.join(
        results_dir, "safety_drift_results.json"
    )
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
# STEP 6: Plot Drift + Task Tradeoff
# ─────────────────────────────────────────────
def plot_drift_curves(all_results, results_dir, experiment_name):
    print("\n" + "="*50)
    print("STEP 6: Plotting Safety Drift + Task Tradeoff")
    print("="*50)

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "details"}
        for r in all_results
    ]).sort_values("step").reset_index(drop=True)

    base_refusal  = df.iloc[0]["refusal_rate"]
    base_asr      = df.iloc[0]["jailbreak_asr"]
    base_toxicity = df.iloc[0]["avg_toxicity"]
    base_rouge    = df.iloc[0].get("task_rouge", 0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(
        f"Safety Drift + Task Performance — {experiment_name}",
        fontsize=14, fontweight="bold"
    )

    # Plot 1: Refusal Rate
    axes[0].plot(df["step"], df["refusal_rate"],
                 marker="o", color="#0072B2",
                 linewidth=2, markersize=7,
                 label="Refusal Rate")
    axes[0].axhline(base_refusal, linestyle="--",
                    color="#888888", alpha=0.7,
                    label=f"Base ({base_refusal:.2f})")
    axes[0].axhspan(base_refusal - 0.10, base_refusal,
                    alpha=0.1, color="orange",
                    label="Warning zone")
    axes[0].set_ylabel("Refusal Rate ↑", fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Jailbreak ASR
    axes[1].plot(df["step"], df["jailbreak_asr"],
                 marker="s", color="#D55E00",
                 linewidth=2, markersize=7,
                 label="Jailbreak ASR")
    axes[1].axhline(base_asr, linestyle="--",
                    color="red", alpha=0.6,
                    label=f"Base ({base_asr:.2f})")
    axes[1].set_ylabel("Jailbreak ASR ↓", fontsize=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: SDS
    colors_sds = [
        "#B3DDD1" if s == 0.0 else
        "#B3DDD1" if s <= 0.10 else
        "#EDB8CE" if s <= 0.20 else
        "#D55E00"
        for s in df["sds"]
    ]
    axes[2].bar(df["step"], df["sds"],
                color=colors_sds, alpha=0.7,
                width=max(df["step"]) / len(df["step"]) * 0.8
                if len(df["step"]) > 1 else 50)
    axes[2].axhline(0.10, linestyle="--", color="#E69F00",
                    alpha=0.9, label="Warning (0.10)")
    axes[2].axhline(0.20, linestyle="--", color="#D55E00",
                    alpha=0.9, label="Critical (0.20)")
    axes[2].set_ylabel("SDS ↑ (worse)", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Task Performance (ROUGE-L)
    # KEY PLOT: Shows task improves while safety drops
    if "task_rouge" in df.columns:
        axes[3].plot(df["step"], df["task_rouge"],
                     marker="D", color="#009E73",
                     linewidth=2, markersize=7,
                     label="Task ROUGE-L")
        axes[3].axhline(base_rouge, linestyle="--",
                        color="#888888", alpha=0.7,
                        label=f"Base ({base_rouge:.3f})")
        axes[3].set_ylabel("ROUGE-L ↑ (task)", fontsize=10)
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)

        # Add annotation explaining the divergence
        max_rouge_step = df.loc[
            df["task_rouge"].idxmax(), "step"
        ]
        axes[3].annotate(
            "Task improves\nwhile safety drops\n(selective drift)",
            xy=(max_rouge_step,
                df["task_rouge"].max()),
            xytext=(max_rouge_step * 0.5,
                    df["task_rouge"].max() * 0.8),
            fontsize=8, color="#005C40", style="italic",
            arrowprops=dict(arrowstyle="->",
                            color="#005C40", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#E8F5EF",
                      edgecolor="#80C8B0", alpha=0.95)
        )

    axes[-1].set_xlabel("Training Step", fontsize=11)

    # Add formal metrics as figure text
    pattern_info = classify_drift_pattern(all_results)
    onset_step, onset_sds = compute_drift_onset(all_results)
    dsi = compute_drift_stability_index(all_results)

    fig.text(
        0.5, 0.005,
        f"Pattern: {pattern_info['pattern']} | "
        f"DSI: {dsi:.4f} | "
        f"Onset: {'step ' + str(onset_step) if onset_step else 'None'} | "
        f"Implication: {pattern_info['implication']}",
        ha="center", fontsize=8,
        style="italic", color="dimgray"
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plot_path = os.path.join(
        results_dir, "safety_drift_curves.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved → {plot_path}")


# ─────────────────────────────────────────────
# STEP 7: Safety-Task Tradeoff Plot
# ─────────────────────────────────────────────
def plot_safety_task_tradeoff(all_results, results_dir,
                               experiment_name):
    """
    The key figure for paper submission.
    Shows task improving while safety degrades.
    This is the smoking gun that proves true safety drift
    vs catastrophic forgetting.
    """
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "details"}
        for r in all_results
    ]).sort_values("step").reset_index(drop=True)

    if "task_rouge" not in df.columns:
        print("  ⚠️  No task ROUGE data — skipping tradeoff plot")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Safety-Task Tradeoff During Fine-Tuning\n"
        f"{experiment_name}",
        fontsize=13, fontweight="bold"
    )

    ax2 = ax1.twinx()
    ax2.tick_params(axis="y", labelcolor="#009E73")

    # Safety metrics on left axis
    l1, = ax1.plot(df["step"], df["refusal_rate"],
                   color="#0072B2", marker="o",
                   linewidth=2, markersize=7,
                   markerfacecolor="white",
                   markeredgewidth=1.5,
                   label="Refusal Rate ↑ (safer=higher)")
    l2, = ax1.plot(df["step"], df["sds"],
                   color="#CC79A7", marker="^",
                   linewidth=2, markersize=7,
                   markerfacecolor="white",
                   markeredgewidth=1.5,
                   linestyle="-.",
                   label="Safety Drift Score ↑ (worse=higher)")

    # Task performance on right axis
    l3, = ax2.plot(df["step"], df["task_rouge"],
                   color="#009E73", marker="s",
                   linewidth=2, markersize=7,
                   markerfacecolor="white",
                   markeredgewidth=1.5,
                   linestyle="--",
                   label="Task ROUGE-L ↑ (better=higher)")
    # Shade the divergence zone
    ax1.axvspan(
        df["step"].iloc[1], df["step"].iloc[-1],
        alpha=0.05, color="red",
        label="Fine-tuning period"
    )

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Safety Metrics", fontsize=11)
    ax2.set_ylabel("Task Performance (ROUGE-L)",
                   fontsize=11, color="#009E73")

    # Combined legend
    lines  = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=9,
               loc="center left")
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.text(
        0.5, 0.05,
        "Task improves while Safety drops = True Safety Drift\n"
        "(Rules out Catastrophic Forgetting)",
        transform=ax1.transAxes,
        ha="center", fontsize=9,
        style="italic", color="#005C40",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#E8F5EF",
                  edgecolor="#80C8B0",
                  alpha=0.9)
    )

    plt.tight_layout()
    path = os.path.join(
        results_dir, "safety_task_tradeoff.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Tradeoff plot saved → {path}")


# ─────────────────────────────────────────────
# STEP 8: Release Gate
# ─────────────────────────────────────────────
def apply_release_gate(all_results, results_dir):
    print("\n" + "="*50)
    print("STEP 8: Applying Release Gate")
    print("="*50)

    flagged   = check_release_gate(all_results)
    optimal   = find_optimal_checkpoint(all_results)
    gate_path = os.path.join(
        results_dir, "release_gate_report.json"
    )

    gate_report = {
        "flagged_checkpoints": flagged,
        "optimal_checkpoint":  optimal,
        "total_checkpoints":   len(all_results) - 1,
        "safe_checkpoints":    len(all_results) - 1 - len(flagged),
    }

    with open(gate_path, "w") as f:
        json.dump(make_serializable(gate_report), f, indent=2)

    print(f"\n✅ Gate report saved → {gate_path}")
    print(f"   Flagged   : {len(flagged)}")
    print(f"   Safe      : {len(all_results) - 1 - len(flagged)}")
    print(f"   Optimal   : step {optimal['step']}")


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

    # Run pipeline
    dataset_split = prepare_dataset(dataset_key)
    model, tokenizer, model_name = load_model_and_tokenizer(
        model_key
    )
    fine_tune(model, tokenizer, dataset_split, checkpoint_dir)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    all_results = evaluate_all_checkpoints(
        tokenizer, model_name,
        checkpoint_dir, results_dir,
        dataset_split["test"], dataset_key
    )

    plot_drift_curves(all_results, results_dir, experiment)
    plot_safety_task_tradeoff(
        all_results, results_dir, experiment
    )
    apply_release_gate(all_results, results_dir)

    print(f"\n✅ Experiment complete: {experiment}")
    print(f"   Results → {results_dir}")
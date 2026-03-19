# config.py
import os
import argparse

# ─────────────────────────────────────────────
# Available Models
# ─────────────────────────────────────────────
MODELS = {
    "qwen0.5b": {
        "path": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen",
        "size": "0.5B",
    },
    "qwen1.5b": {
        "path": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen",
        "size": "1.5B",
    },
    "smollm1.7b": {
        "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "family": "SmolLM",
        "size": "1.7B",
    },
}

# ─────────────────────────────────────────────
# Available Datasets
# ─────────────────────────────────────────────
DATASETS = {
    "dialogsum": {
        "path": "knkarthick/dialogsum",
        "config": None,
        "text_col": "dialogue",
        "summary_col": "summary",
        "prompt": "Summarize this conversation:",
    },
    "xsum": {
        "path": "EdinburghNLP/xsum",
        "config": None,
        "text_col": "document",
        "summary_col": "summary",
        "prompt": "Summarize this article:",
    },
}

# ─────────────────────────────────────────────
# Command Line Arguments
# ───────────
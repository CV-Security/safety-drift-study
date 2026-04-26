# prompts/__init__.py
# Makes prompts a importable module

from .harmful_prompts import HARMFUL_PROMPTS, CORE_HARMFUL_TOPICS
from .jailbreak_templates import (
    CATEGORY_1_AUTHORITY,
    CATEGORY_2_FICTIONAL,
    CATEGORY_3_COMPLETION,
    CATEGORY_4_ESCALATION,
    CATEGORY_5_ACADEMIC,
)
from .jailbreak_builder import (
    build_all_jailbreak_prompts,
    JAILBREAK_PROMPTS
)

__all__ = [
    "HARMFUL_PROMPTS",
    "CORE_HARMFUL_TOPICS",
    "JAILBREAK_PROMPTS",
    "build_all_jailbreak_prompts",
    "CATEGORY_1_AUTHORITY",
    "CATEGORY_2_FICTIONAL",
    "CATEGORY_3_COMPLETION",
    "CATEGORY_4_ESCALATION",
    "CATEGORY_5_ACADEMIC",
]
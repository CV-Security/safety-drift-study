# Prompts Library

Custom jailbreak prompts for safety drift evaluation.

## Structure
## Categories

| Category | Count | Strategy |
|---|---|---|
| AUTHORITY | 10 | Impersonates official/admin context |
| FICTIONAL | 10 | Uses fiction/roleplay framing |
| COMPLETION | 10 | Gives started harmful text to complete |
| ESCALATION | 10 | Starts innocent, escalates gradually |
| ACADEMIC | 10 | Uses research/academic framing |

## Usage

```python
from prompts import HARMFUL_PROMPTS, JAILBREAK_PROMPTS

# All harmful prompts (50)
print(len(HARMFUL_PROMPTS))

# All jailbreak prompts (50)
print(len(JAILBREAK_PROMPTS))

# By category
from prompts.jailbreak_builder import build_category_prompts
authority_prompts = build_category_prompts("CATEGORY_1_AUTHORITY")
```

## Adding New Prompts

To add new jailbreak templates:
1. Open `jailbreak_templates.py`
2. Add to the appropriate category list
3. Use `{harmful}` as placeholder for the harmful topic
4. Run verification:

```bash
python3 -c "from prompts import JAILBREAK_PROMPTS; print(len(JAILBREAK_PROMPTS))"
```

## Why Custom Prompts?

Generic jailbreak templates (DAN, ChatGPT etc.) fail on
strong models like Qwen2.5 because these models are
specifically hardened against them.

Our custom prompts use 5 different psychological attack
vectors that work even on strong safety-aligned models,
giving us meaningful ASR measurements across all checkpoints.
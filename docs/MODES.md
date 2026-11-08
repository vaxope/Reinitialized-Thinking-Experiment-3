# Reasoning modes: exact definitions

Source: Qwen3.5-2B `chat_template.jinja`, generation-prompt block:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

**Consequence:** thinking is **off unless `enable_thinking=True` is passed explicitly.** A tool or config that never sets it runs every "think" experiment as no-think. Always set it, and log the full input text.

## The four modes

A user turn is `<|im_start|>user\n{user_text}<|im_end|>\n`, possibly preceded by a system turn. Only the user text and the start of the assistant turn differ between modes:

### think
```
<|im_start|>user
{QUESTION_PROMPT}<|im_end|>
<|im_start|>assistant
<think>
```
The model writes its hidden reasoning, closes `</think>`, then writes the answer.

### nothink
```
<|im_start|>user
{QUESTION_PROMPT}<|im_end|>
<|im_start|>assistant
<think>

</think>

```
Hidden thinking is skipped. **The model may still reason in its visible answer.**

### direct
Same assistant prefix as `nothink`. The user text adds: *"Respond with only the final answer line, with no explanation."* Generation is capped at 16 new tokens, so no room for reasoning. This measures what the model knows without reasoning.

### gar (reinitialized thinking)
Same assistant prefix as `nothink`. The user text adds GAR-style analysis steps (see `src/modes.py`, `GAR_STEPS`): reason inside `<analysis>…</analysis>` following the numbered steps, then give the answer line.

## Notes
- GAR's own code (for Qwen3) appends `"\n<think>\n\n</think>\n"` itself, with different whitespace from Qwen3.5's template. Whitespace changes token IDs, so this project uses the **official template's** exact string.
- `QUESTION_PROMPT` is identical across modes except for the mode-specific instruction. Options are shuffled with a seed fixed per (question, sample) and shared by all modes.
- Required final line in every mode: `Answer: X`, where X is one of the valid letters.

"""The generation-record schema: the contract between generation and scoring.

Purpose : one list of field names (from CLAUDE.md "Logging") that every record in
          runs/<run>/generations.jsonl must have. src/score.py checks records against
          it; the future src/generate.py will write exactly these fields.
Pipeline: (later) generate.py -> generations.jsonl -> [schema check in score.py] -> scored.jsonl
"""

# Written at generation time (before any scoring). Order follows CLAUDE.md.
GENERATION_FIELDS = (
    "run_id", "mode", "model", "model_revision", "qid", "sample_idx", "seed",
    "option_order", "chat_template_hash", "input_text", "input_token_ids",
    "output_text", "output_token_ids", "output_logprobs", "finish_reason",
    "n_input_tokens", "n_output_tokens", "n_think_tokens", "gold",
    "sampling_params", "sglang_version", "wall_time_s", "error",
)

# Filled in only by src/score.py, never by generation. In generations.jsonl they are
# either absent or null; scoring writes them into a separate scored.jsonl.
SCORE_FIELDS = ("extracted_answer", "extraction_rule", "correct")

# Extra fields score.py adds for transparency (not in CLAUDE.md's list, purely additive).
SCORE_EXTRA_FIELDS = ("extraction_detail", "outcome", "truncated", "scoring_rule_version")


def missing_fields(record: dict) -> list[str]:
    """Generation fields absent from a record (null values are allowed: e.g. error=None)."""
    return [k for k in GENERATION_FIELDS if k not in record]

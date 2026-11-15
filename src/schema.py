"""The generation-record schema: the contract between generation and scoring.

Purpose : one list of field names (from CLAUDE.md "Logging") that every record in
          runs/<run>/generations.jsonl must have. src/score.py checks records against
          it; src/generate.py writes exactly these fields (plus the extras below).
Pipeline: generate.py -> generations.jsonl -> [schema check in score.py] -> scored.jsonl
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

# Extra fields src/generate.py writes (not in CLAUDE.md's list, purely additive):
GENERATE_EXTRA_FIELDS = (
    "prompt_hash",            # hash of our prompt template (src/prompts.PROMPT_HASH)
    "slurm_job_id",           # which Slurm job produced the record
    "think_closed",           # think mode: did </think> appear in the output ids
    "n_analysis_tokens",      # reinit mode: tokens up to and including the first </analysis>
                              #   (all tokens if it never closed); 0 in the other modes
    "analysis_closed",        # reinit mode: did </analysis> appear in the output
    "request_started_at",     # UTC time the request was sent
    "server_text",            # SGLang's own decoded text, ONLY if it differs from ours
    "sglang_meta",            # SGLang's meta_info (finish reason detail, token counts, latency)
    "response_checks",        # ids source; server text / token counts agree with ours
    "error_traceback",        # only on errors
)

# Extra fields score.py adds for transparency (not in CLAUDE.md's list, purely additive).
SCORE_EXTRA_FIELDS = ("extraction_detail", "outcome", "truncated", "scoring_rule_version")


def missing_fields(record: dict) -> list[str]:
    """Generation fields absent from a record (null values are allowed: e.g. error=None)."""
    return [k for k in GENERATION_FIELDS if k not in record]

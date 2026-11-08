# Single source of truth for training parameters (Phase A: replicate GAR's settings for the solver).
# Every value: what it does | GAR official value | why this value. See docs/PARAMETERS.md.
# Flag names verified against slime @79aa41a (slime/utils/arguments.py) unless marked MEGATRON.
# Changing any value after a run starts requires a new run name and a vault decision note.

RUN_NAME=${RUN_NAME:-gar2b_phaseA}
MODE=${MODE:-gar}                      # think | nothink | direct | gar  (see docs/MODES.md)
ENABLE_THINKING=$([ "$MODE" = "think" ] && echo true || echo false)

DATA_ARGS=(
   --prompt-data "${DATA_DIR}/train_${MODE}.jsonl"   # built by the data-prep component
   --input-key messages                 # chat messages; the GAR steps are already in the user text
   --label-key label                    # gold letter
   --metadata-key metadata              # qid, n_options, option order, source
   --apply-chat-template                # slime formats with the model's own template...
   --apply-chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}"  # ...with thinking set EXPLICITLY (Qwen3.5 default is off)
   --rollout-shuffle                    # random prompt order (GAR sorted training data by length; we don't)
)

ROLLOUT_ARGS=(
   --n-samples-per-prompt 16            # group size G | GAR: 16 proofs per theorem
   --rollout-batch-size 16              # prompts per rollout -> 16 x 16 = 256 samples per rollout
   --global-batch-size 256              # samples per optimizer step -> 1 step per rollout (fully on-policy)
                                        # GAR: global batch 32, one epoch over each round's data (many off-policy steps)
   --num-rollout 200                    # training length: 200 rollouts x 16 prompts = 3,200 prompts
   --rollout-max-prompt-len 2048        # longest formatted question must fit; data prep checks this
   --rollout-max-response-len 16384     # generation cap | GAR prover: 16,384. Avoids the 4,096 truncation trap
   --rollout-temperature 1.0            # GAR prover: 1.0
   --rollout-top-p 1.0                  # GAR: 0.95. 1.0 keeps rollout and training probabilities consistent
   --rollout-seed 42
)

REWARD_ARGS=(
   --custom-rm-path src.reward.custom_rm   # strict last "Answer: X" line; see src/reward.py for why not --rm-type gpqa
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
                                        # drop groups where all 16 answers score the same (no learning signal)
                                        # GAR additionally drops pass rate > 0.5; add as a Phase B variant
   --over-sampling-batch-size 32        # sample extra prompts so filtering still fills the batch
)

GRPO_ARGS=(
   --advantage-estimator grpo           # (r - group mean) / group std | GAR: same, per theorem
   --eps-clip 0.2                       # PPO clip range | GAR: 0.2 symmetric
   --use-kl-loss                        # KL penalty toward the starting model | GAR: yes
   --kl-loss-coef 0.01                  # | GAR beta_kl: 0.01
   --kl-loss-type k3                    # | GAR: k3 estimator exp(ref-θ) - (ref-θ) - 1
   --entropy-coef 0.0
   # Loss reduction: default (mean per sequence, then over batch) matches GAR.
   # Phase B: --calculate-per-token-loss, --eps-clip-high 0.28, --disable-grpo-std-normalization
)

OPTIMIZER_ARGS=(
   --optimizer adam                     # MEGATRON (Adam with decoupled weight decay)
   --lr 1e-6                            # GAR: 5e-6 for 8B models. 1e-6 is slime's default and safer for 2B full fine-tuning
   --lr-decay-style constant            # MEGATRON | GAR: constant, no warmup
   --weight-decay 1e-8                  # MEGATRON | GAR: 1e-8 (effectively none)
   --adam-beta1 0.9                     # MEGATRON | GAR: PyTorch AdamW defaults (0.9, 0.999)
   --adam-beta2 0.999
   --clip-grad 1.0                      # gradient-norm clip | GAR: none found. Kept as a safety net; logged
   --seed 1234
)

LOG_ARGS=(
   --save-debug-rollout-data "${RUN_DIR}/rollouts/rollout_{rollout_id}.pt"   # every trajectory, every rollout
   --eval-interval 10
   --eval-prompt-data heldout "${DATA_DIR}/heldout_${MODE}.jsonl"   # held-out split of the TRAINING source, never GPQA
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 16384
   --eval-temperature 1.0
   --eval-top-p 1.0
   --save-interval 20
)

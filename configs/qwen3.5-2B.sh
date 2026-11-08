# Megatron model arguments for Qwen3.5-2B, written for this project.
# slime @79aa41a ships qwen3.5-0.8B.sh and qwen3.5-4B.sh but no 2B script. This file follows
# their structure, with sizes taken from Qwen/Qwen3.5-2B config.json (text_config):
#   hidden_size 2048, num_hidden_layers 24, num_attention_heads 8, num_key_value_heads 2,
#   head_dim 256, intermediate_size 6144, vocab_size 248320, rope_theta 1e7,
#   partial_rotary_factor 0.25, rms_norm_eps 1e-6, tie_word_embeddings true,
#   layer pattern: 3 linear-attention (gated delta) layers then 1 full-attention layer.
# The linear-attention settings are read by slime's qwen3_5 spec plugin, as in the 0.8B/4B scripts.
# VERIFY with gate 4 (HF -> torch_dist -> HF round trip, compare logits) before any training.
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 8
   --num-query-groups 2
   --kv-channels 256
   --num-layers 24
   --hidden-size 2048
   --ffn-hidden-size 6144
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)

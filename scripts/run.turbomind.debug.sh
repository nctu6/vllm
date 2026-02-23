VLLM_TURBOMIND_STRICT_LMDEPLOY_STOP=1 \
VLLM_TURBOMIND_NO_AUTO_RANDOM_SEED=1 \
VLLM_TURBOMIND_FORCE_GREEDY=1 \
VLLM_TURBOMIND_DEBUG_OUTPUT_LOGPROBS=20 \
VLLM_TURBOMIND_DEBUG=1 \
CUDA_VISIBLE_DEVICES=4 \
vllm serve /models/Qwen/Qwen3-8B \
  --served-model-name test \
  --host 0.0.0.0 \
  --port 8976 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype float16 \
  --backend turbomind 2>&1 | tee run.log


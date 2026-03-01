CUDA_VISIBLE_DEVICES=5 \
vllm serve /models/Qwen/Qwen3-8B \
  --served-model-name test \
  --host 0.0.0.0 \
  --port 8976 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype float16 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"prompt_lookup_min":3,"prompt_lookup_max":8}' \
  --backend vllm 2>&1 | tee run.log


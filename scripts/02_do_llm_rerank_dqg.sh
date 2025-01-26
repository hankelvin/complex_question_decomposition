conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python llm_inference/chat_class_inference.py --task "rerank_dqg_"$1  --model_name $2 --input_file $3  $4 --model_size $5 --rerank_dqg_models $6 --hf_token $7 
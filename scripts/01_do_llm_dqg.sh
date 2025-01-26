conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python llm_inference/chat_class_inference.py --task "decomp_qg"  --model_name $1 --input_file $2  $3 --model_size $4 --decomp_qg_args $5 $6 --hf_token $7
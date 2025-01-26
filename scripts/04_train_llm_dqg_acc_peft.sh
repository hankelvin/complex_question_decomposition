conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python dqg_dqa/train_inference.py --model_name $1 --task "dqg_"$2 --use_dqg_llmgen $3 --test_only $4 --use_dqg_rtfilt $5 --decomp_qg_args $6 --hf_token $7
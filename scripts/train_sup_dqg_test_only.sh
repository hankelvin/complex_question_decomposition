conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python dqg_dqa/train_inference.py --task $1 --test_only $2 --cross_domain $3
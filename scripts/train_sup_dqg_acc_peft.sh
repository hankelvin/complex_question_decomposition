conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 python dqg_dqa/train_inference.py --task $1 --resume_ckpt_path $2 --model $3  --do_lora True --use_accelerate True --hf_token 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
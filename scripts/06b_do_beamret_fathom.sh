conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:16000  python tools/beam_retriever/test_model_SQ.py --model_type $1 --dataset_type $2 --test_ckpt_path "../../"$3 --is_dev $4 --use_gold_passages $5 --decomp_origin $6 --output_variants $7
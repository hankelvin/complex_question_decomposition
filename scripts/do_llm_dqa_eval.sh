conda deactivate
conda activate decompqg

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:16000 python dqg_dqa/train_inference.py --task "dqa_"$1 --test_only phase3_val_as_test --do_lora False --model_name $2 --do_dqg_llmgen_qa $3 --do_dqg_llmgen_qa_usecontext $4 --hf_token $5
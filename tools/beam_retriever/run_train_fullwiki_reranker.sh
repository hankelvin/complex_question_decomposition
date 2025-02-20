conda deactivate
conda activate llama

# CUDA_VISIBLE_DEVICES=2 \
python train_beam_retriever.py \
--do_train \
--gradient_checkpointing \
--prefix \
retr_hotpot_fullwiki_v2_50_beam_size1_base \
--model_name \
models/deberta-v3-base \
--tokenizer_path \
models/deberta-v3-base \
--dataset_type \
hotpot_reranker \
--train_file \
data/hotpotqa/hotpotqa_fullwiki_br_train_v2_50.json \
--predict_file \
data/hotpotqa/hotpotqa_fullwiki_br_dev_v2_50.json \
--train_batch_size \
8 \
--learning_rate \
2e-5 \
--fp16 \
--beam_size \
1 \
--predict_batch_size \
1 \
--warmup-ratio \
0.1 \
--num_train_epochs \
20 \
--mean_passage_len \
250 \
--log_period_ratio \
0.01 \
--accumulate_gradients \
8 \
--eval_period_ratio \
0.3
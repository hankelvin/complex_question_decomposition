conda deactivate
conda activate llama

# CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 \
python train_beam_retriever.py \
--do_train \
--gradient_checkpointing \
--prefix \
deberta_use_two_classier_musique_beam_size2 \
--model_name \
microsoft/deberta-v3-base \
--tokenizer_path \
microsoft/deberta-v3-base \
--dataset_type \
musique \
--train_file \
../../data/musique/musique_ans_v1.0_train.jsonl \
--predict_file \
../../data/musique/musique_ans_v1.0_dev.jsonl \
--train_batch_size \
32 \
--learning_rate \
2e-5 \
--mean_passage_len \
120 \
--fp16 \
--beam_size \
2 \
--predict_batch_size \
1 \
--num_train_epochs \
16 \
--warmup-ratio \
0.1 \
--log_period_ratio \
0.01 \
--accumulate_gradients \
8 \
--eval_period_ratio \
1.0 \
--do_single_hop \
$1 \
# --init_checkpoint \
# ''
conda deactivate
conda activate llama

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20000 \
python tools/beam_retriever/train_reader.py \
--do_train \
--prefix \
musique_reader_deberta_large \
--model_name \
microsoft/deberta-v3-large \
--tokenizer_path \
microsoft/deberta-v3-large \
--dataset_type \
musique \
--train_file \
"data/musique/musique_ans_v1.0_train.jsonl" \
--predict_file \
"data/musique/musique_ans_v1.0_dev.jsonl" \
--train_batch_size \
6 \
--learning_rate \
5e-6 \
--fp16 \
--max_seq_len \
1024 \
--num_train_epochs \
12 \
--predict_batch_size \
32 \
--warmup-ratio \
0.1 \
--log_period_ratio \
0.01 \
--eval_period_ratio \
1.0 \
--do_single_hop \
$1 \
--sq_all_paragraphs \
$2 \
# --init_checkpoint \
# ''
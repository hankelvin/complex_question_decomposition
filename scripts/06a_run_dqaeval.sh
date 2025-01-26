conda deactivate
conda activate decompqg

./scripts/do_llm_dqa_eval.sh  $1 $2 "gpt4o"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "sft_ft5"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "llm_single_llama"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "llm_single_qwen"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "llm_single_phi3"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "llm_single_gemma"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "llm_top1_sft_llama"$3 True $4
./scripts/do_llm_dqa_eval.sh  $1 $2 "llm_top1_sft_qwen"$3 True $4

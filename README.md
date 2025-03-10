# Generating complex questions decompositions in the face of distribution shifts
This repository holds the data, code and models in the NAACL 2025 paper, [_"Generating complex questions decompositions in the face of distribution shifts"_](https://github.com/hankelvin/complex_question_decomposition/blob/main/paper/naacl25_camera_ready.pdf).

## Preliminaries
Install requirements:

-- conda environment for LLM inference, finetuning
```
conda create --name=decompqg python=3.10 -y
conda activate decompqg
CUDA="{}"       # your available CUDA version (nvcc -v)
CUDA_TORCH=${CUDA//./} # remove "."
HFTOKEN="{}"    # your huggingface token for accesssing gated models (see https://huggingface.co/docs/hub/security-tokens)
conda install -c "nvidia/label/cuda-"$CUDA cuda-toolkit -y
python -m pip install torch==2.4.1 --index-url "https://download.pytorch.org/whl/cu"$CUDA_TORCH
python -m pip install -r requirements.txt
```
-- conda environment for STV and automatic metrics
```
conda deactivate
conda create --name=breakeval python=3.7 -y
conda activate breakeval
python -m pip install -r requirements_breakeval.txt
python -m pip install sacrebleu datasets nltk pyrankvote==2.0.6
python -m spacy download en_core_web_sm
```

Data: 
The data used in our experiments can found [here](https://drive.google.com/drive/folders/1zCZYtx9pw3Uzh4KVf6HWRtwY7S4W3WsY?usp=sharing). Unzip ```data_files.zip``` and move its contents into the ```data``` folder in this repository.

---
## To replicate the results in the paper, do the following:

### 1. generating question decomposition candidates
```
conda activate decompqg
./scripts/01_do_llm_dqg.sh  ge breakhigh   validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  ll breakhigh   validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  ph breakhigh   validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  qw breakhigh   validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  ge musique     validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  ll musique     validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  ph musique     validation 'normal' 5 True $HFTOKEN
./scripts/01_do_llm_dqg.sh  qw musique     validation 'normal' 5 True $HFTOKEN
```
---
### 2. obtain ranked preferences for question decompositions
```
conda activate decompqg
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ay breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ge breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ll breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise mi breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise nv breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ol breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ph breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise qw breakhigh  validation 'normal' 'ge ll ph qw' $HFTOKEN

./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ay musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ge musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ll musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise mi musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise nv musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ol musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise ph musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
./scripts/02_do_llm_rerank_dqg.sh breakhigh_listwise qw musique    validation 'normal' 'ge ll ph qw' $HFTOKEN
```
---
### 3. run STV and pick preferred decomposition candidate
```
conda deactivate
conda activate breakeval
# for 4x CLLMs
python evaluation/evaluate_stv.py --dataset 1
python evaluation/evaluate_stv.py --dataset 2
# for diff 4x CLLMs
python evaluation/evaluate_stv.py --dataset 1 --rankllm_systems "aya mistral nvidia_llama3 olmo"
python evaluation/evaluate_stv.py --dataset 2 --rankllm_systems "aya mistral nvidia_llama3 olmo"
# for 8x CLLMs 
python evaluation/evaluate_stv.py --dataset 1 --rankllm_systems "aya gemma llama mistral nvidia_llama3 olmo phi3 qwen"
python evaluation/evaluate_stv.py --dataset 2 --rankllm_systems "aya gemma llama mistral nvidia_llama3 olmo phi3 qwen"
conda deactivate
```
---
### 4. fine-tune CLLMs, obtain predicted decompositions
#### a. finetuning
```
conda activate decompqg
./scripts/04_train_llm_dqg_acc_peft.sh llama  	breakhigh   'None' False 'None' '3 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	breakhigh   'None' False 'None' '3 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh llama  	musique     'None' False 'None' '3 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	musique     'None' False 'None' '3 True' $HFTOKEN
```

#### b. obtaining predictions
##### Models 
Our fine-tuned models can found [here](https://drive.google.com/drive/folders/1zCZYtx9pw3Uzh4KVf6HWRtwY7S4W3WsY?usp=sharing). Unzip ```models.zip``` and  move the files into this repo by following the directory structure in there,. NOTE: The __FT-PANEL__ models were trained with LoRA adapters and this folder only holds the adapter weights. If you would like to, you can modify the scripts here to load the models with [PEFT](https://github.com/huggingface/peft) and merge.

```
conda activate decompqg
./scripts/04_train_llm_dqg_acc_peft.sh llama  	breakhigh   'None' phase3_val_as_test 'None' '3 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	breakhigh   'None' phase3_val_as_test 'None' '3 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh llama  	musique     'None' phase3_val_as_test 'None' '3 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	musique     'None' phase3_val_as_test 'None' '3 True' $HFTOKEN
```
---
### 5. run automatic metrics: exact match, SARI and GED
```
conda deactivate
conda activate breakeval
python evaluation/evaluate_break_suite.py
conda deactivate
```

### 6. run QA evaluation 
#### a. using Llama 3.1 8B & Qwen 2.5 7B
```
conda activate decompqg
./scripts/06a_run_dqaeval.sh breakhigh    llama   "" $HFTOKEN
./scripts/06a_run_dqaeval.sh breakhigh    qwen    "" $HFTOKEN 
./scripts/06a_run_dqaeval.sh musique      llama   "" $HFTOKEN
./scripts/06a_run_dqaeval.sh musique      qwen    "" $HFTOKEN 
conda deactivate
```

### b. using supervised QA model (for musique only)
-- to train the QA model
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:16000 ./tools/beam_retriever/run_train_reader_musique.sh True True
```

-- to run QA
```
conda activate decompqg
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'gpt4o None musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'supervised musique musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_single gemma musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_single llama musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_single phi3 musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_single qwen musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_top1 llama musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_top1 qwen musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_top1_sft llama musique' None 
./scripts/06b_do_beamret_fathom.sh musique musique None True True 'llm_top1_sft qwen musique' None 
```

### 7. cross-domain experiments
#### a. obtaining predictions
```
conda activate decompqg
./scripts/07a_train_llm_dqg_acc_peft_CROSSDOMAINTEST.sh llama  	breakhigh   'None' phase3_val_as_test 'None' '5 True' musique   $HFTOKEN
./scripts/07a_train_llm_dqg_acc_peft_CROSSDOMAINTEST.sh qwen  	breakhigh   'None' phase3_val_as_test 'None' '5 True' musique   $HFTOKEN
./scripts/07a_train_llm_dqg_acc_peft_CROSSDOMAINTEST.sh llama  	musique     'None' phase3_val_as_test 'None' '5 True' breakhigh $HFTOKEN
./scripts/07a_train_llm_dqg_acc_peft_CROSSDOMAINTEST.sh qwen  	musique     'None' phase3_val_as_test 'None' '5 True' breakhigh $HFTOKEN
```

#### b. QA evaluation with Llama & Qwen (public checkpoints, not fine-tuned)
```
conda activate decompqg
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  breakhigh llama llm_top1_sft_crossdomain_llama True musique   $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  breakhigh qwen  llm_top1_sft_crossdomain_llama True musique   $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  breakhigh llama llm_top1_sft_crossdomain_qwen  True musique   $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  breakhigh qwen  llm_top1_sft_crossdomain_qwen  True musique   $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  musique   llama llm_top1_sft_crossdomain_llama True breakhigh $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  musique   qwen  llm_top1_sft_crossdomain_llama True breakhigh $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  musique   llama llm_top1_sft_crossdomain_qwen  True breakhigh $HFTOKEN
./scripts/07a_do_llm_dqa_eval_CROSSDOMAINTEST.sh  musique   qwen  llm_top1_sft_crossdomain_qwen  True breakhigh $HFTOKEN
```

## Citation
If you find our work useful, please cite our publication:

```
@inproceedings{han-gardent-2025-generating,
    title = "Generating complex question decompositions in the face of distribution shifts",
    author = "Han, Kelvin and Gardent, Claire",
    booktitle = "Proceedings of the 2025 Conference of the Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico, USA",
    publisher = "Association for Computational Linguistics",}
```
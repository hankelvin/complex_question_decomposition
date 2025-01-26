# Generating complex questions decompositions in the face of distribution shifts

### 0. preliminaries
Install requirements:

-- for LLM inference, finetuning
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
-- for STV and automatic metrics
```
conda deactivate
conda create --name=breakeval python=3.7 -y
conda activate breakeval
python -m pip install -r requirements_breakeval.txt
python -m pip install sacrebleu datasets nltk pyrankvote==2.0.6
python -m spacy download en_core_web_sm
```

---

### To replicate the results in the paper, do the following:
---
### 1. generating question decomposition candidates
```
conda activate decompqg
./scripts/01_do_llm_dqg.sh  ge breakhigh   validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  ll breakhigh   validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  ph breakhigh   validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  qw breakhigh   validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  ge musique     validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  ll musique     validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  ph musique     validation 'normal' 0 False $HFTOKEN
./scripts/01_do_llm_dqg.sh  qw musique     validation 'normal' 0 False $HFTOKEN
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
./scripts/04_train_llm_dqg_acc_peft.sh llama  	breakhigh   'None' False 'None' '0 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	breakhigh   'None' False 'None' '0 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh llama  	musique     'None' False 'None' '0 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	musique     'None' False 'None' '0 True' $HFTOKEN
```

#### b. obtaining predictions
```
conda activate decompqg
./scripts/04_train_llm_dqg_acc_peft.sh llama  	breakhigh   'None' phase3_val_as_test 'None' '0 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	breakhigh   'None' phase3_val_as_test 'None' '0 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh llama  	musique     'None' phase3_val_as_test 'None' '0 True' $HFTOKEN
./scripts/04_train_llm_dqg_acc_peft.sh qwen  	musique     'None' phase3_val_as_test 'None' '0 True' $HFTOKEN
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
#### a. using Llama 3.1 8B
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
./scripts/do_beamret_fathom.sh musique musique None True True 'gpt4o None musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'supervised musique musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_single gemma musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_single llama musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_single phi3 musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_single qwen musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_top1 llama musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_top1 qwen musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_top1_sft llama musique' None 
./scripts/do_beamret_fathom.sh musique musique None True True 'llm_top1_sft qwen musique' None 
```

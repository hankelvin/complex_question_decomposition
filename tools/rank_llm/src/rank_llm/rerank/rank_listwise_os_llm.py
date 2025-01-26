import json, re, random, numpy as np
from typing import Optional, Tuple

import torch
from fastchat.model import get_conversation_template, load_model
from ftfy import fix_text
from transformers.generation import GenerationConfig

import sys, os
currpath = os.getcwd()
# get file path
filepath = os.path.realpath(__file__)
# change dir 
os.chdir(os.path.dirname(filepath))
sys.path.append('../../rank_llm')
from rerank.rankllm import PromptMode, RankLLM
from data import Result
os.chdir(currpath)


class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        ### CHANGE START ###
        bypass_fsc_load: bool = False,
        hf_model = None,
        tokenizer = None,
        rerank_task_name = None,
        rerank_dqg_tasks = [],
        constraints_dict = None,
        rerank_with_score_scheme = False,
        qpos_tokens = None,
        prompt_version = 1, 
        ### CHANGE END ###
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using
         a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage
         handling, and custom system messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - variable_passages (bool, optional): Indicates whether the number of passages to rank can vary. Defaults to False.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - system_message (Optional[str], optional): Custom system message to be included in the prompt for additional
         instructions or context. Defaults to None.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        ### CHANGE START ###
        self.bypass_fsc_load        = bypass_fsc_load
        self.model_path             = model
        self.rerank_task_name       = rerank_task_name
        self.rerank_dqg_tasks       = rerank_dqg_tasks
        self.qpos_tokens            = qpos_tokens
        self.input_context_cands_post = "Complex Multi-hop Query: '{}'.\n\nDecomposition Attempts: \n"
        for k,v in constraints_dict.items():
            setattr(self, k, v) # self.constraints, self.num_beams
        self.rerank_with_score_scheme = rerank_with_score_scheme
        self.prompt_version           = prompt_version
        ### CHANGE END ###
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )
        # ToDo: Make repetition_penalty configurable
        ### CHANGE START ###
        if bypass_fsc_load:
            assert hf_model is not None and tokenizer is not None
            self._llm, self._tokenizer = hf_model, tokenizer
        else: self._llm, self._tokenizer = load_model(model, device=device, num_gpus=num_gpus)
        ### CHANGE END ###
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        if num_few_shot_examples > 0:
            if self.rerank_task_name in self.rerank_dqg_tasks:
                dataset = re.search(r'musique|breakhigh|2wikimultihop', self.rerank_task_name).group(0)
                if not dataset: raise NotImplementedError
                dp = 'tools/rank_llm/src/rank_llm/rerank'
                with open(f"{dp}/rerank_dqg_examples.json", "r", encoding = 'utf-8') as json_file:
                    self._examples = json.load(json_file)[dataset]

            else:
                with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                    self._examples = list(json_file)[1:-1]

    def run_llm(
        self, prompt: str, 
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        if current_window_size is None:
            current_window_size = self._window_size
        inputs = self._tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size)
        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False
        ### CHANGE START ### for Llama models (eos_token_id needs to be set to terminators)
        gen_args = {}
        if self.bypass_fsc_load:
            c1_check_str = any(n in self.model_path for n in ['Meta-Llama-3', 'Llama-3'])
            c2_check_str = any(n in self.model_path for n in ['GritLM/GritLM-7B'])
            c3_check_str = any(n in self.model_path for n in ['mistralai/Mistral-7B-Instruct-v0.3',
                                                              'mistralai/Mistral-7B-Instruct-v0.3',
                                                              'Qwen/Qwen1.5-7B-Chat', 
                                                              'Qwen/Qwen2.5-7B-Instruct', 
                                                              'nvidia/Llama3-ChatQA-1.5-8B'])
            if c1_check_str:
                terminators = [self._tokenizer.eos_token_id,
                               self._tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                gen_args = {'eos_token_id': terminators, 
                              # in llama, the pad_token_id is not set (rec: use the eos_token_id)
                              'pad_token_id': self._tokenizer.eos_token_id,}
            elif c2_check_str:
                gen_args = {'pad_token_id': self._tokenizer.pad_token_id,}
            elif c3_check_str:
                # in mistral, the pad_token_id is not set (rec: use the eos_token_id)
                gen_args = {'pad_token_id': self._tokenizer.eos_token_id,}
        
        # gen_args['max_new_tokens']          = self.max_new_tokens # already set in gen_cfg.max_new_tokens
        gen_args['return_dict_in_generate'] = True
        if self.constraints is not None:  gen_args['constraints']   = self.constraints
        if self.num_beams is not None:    gen_args['num_beams']     = self.num_beams
        
        ### CHANGE START ###
        # a. use generate to obtain the ranker response
        output_holder = self._llm.generate(**inputs, **gen_args, generation_config = gen_cfg)

        if gen_args['return_dict_in_generate']: 
            output_ids = output_holder['sequences']
        else: output_ids = output_holder
        ### CHANGE END ###

        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        
        del output_holder
        return outputs, output_ids.size(0)
        ### CHANGE END ###

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            ### CHANGE START ### zfill(2)
            seq     = " > ".join([f"[{str(i+1).zfill(2)}]" for i in range(current_window_size)])
            seq_tok = self._tokenizer.encode(seq) # to be safe, round-around check instead of skip_special_tokens=True
            if self._tokenizer.bos_token_id is not None:
                if seq_tok[0] == self._tokenizer.bos_token_id: seq_tok = seq_tok[1:]
            if self._tokenizer.eos_token_id is not None:
                if seq_tok[-1] == self._tokenizer.eos_token_id: seq_tok = seq_tok[:-1]
            # _output_token_estimate = (
            #     len(
            #         self._tokenizer.encode(seq)
            #     )
            #     - 1
            # )
            _output_token_estimate = len(seq_tok)
            ### CHANGE END ### 
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def _add_prefix_prompt(self, query: str, num: int, stance: str = None) -> str:
        ### CHANGE START ###
        if self.rerank_task_name in self.rerank_dqg_tasks:
            if self.prompt_version == 1:
                if self.rerank_with_score_scheme:
                    prefix_prompt = f"I will provide you with {num} attempts that were made at decomposing a complex multi-hop question into sequences of simpler sub-questions. Each attempt is indicated by a numerical identifier [] in front of it. Rank the attempts based on their quality with respect to the complex multi-hop question. You should rank the attempts by scoring each of them using the following criteria (the maximum attainable score by an attempt is 10 points): \n(i) the sub-questions are grammatically sound (1 point), \n(ii) the sequence of sub-questions does not contain superfluous steps that are unnecessary for answering the complex multi-hop question (3 points), \n(iii) each sub-question in the sequence does not logically contradict the question(s) that comes before it (3 points), and \n(iv) as a whole, the sub-questions are not ambiguously worded and cover all the necessary information and steps that will allow us to arrive at the same answer as the original complex question (3 points).\n"
                
                else: 
                    prefix_prompt = f"I will provide you with {num} attempts that were made at decomposing a complex multi-hop question into sequences of simpler sub-questions. Each attempt is indicated by a numerical identifier [] in front of it. Rank the attempts based on their quality with respect to the complex multi-hop question. You should rank the attempts based on the following criteria: \n(i) the sub-questions are grammatically sound, \n(ii) the sequence of sub-questions do not contain superfluous steps that are unnecessary for answering the complex multi-hop question, \n(iii) each sub-question in the sequence does not logically contradict the question(s) that comes before it, and \n(iv) as a whole, the sub-questions are not ambiguously worded and cover all the necessary information and steps that will allow us to arrive at the same answer as the original complex question.\n"

            elif self.prompt_version == 2:
                if self.rerank_with_score_scheme:
                    prefix_prompt = f"I will provide you with {num} attempts that were made at decomposing a complex multi-hop question into a sequence of simpler sub-questions. Each attempt is indicated by a numerical identifier [] in front of it. Rank the attempts based on their quality with respect to the complex multi-hop question. You should rank the attempts by scoring each of them using the following criteria (the maximum attainable score by an attempt is 10 points): \n(i) the sub-questions are grammatically sound (1 point), \n(ii) the sub-questions are not ambiguously phrased or overly general such that it becomes difficult to find their answers (2 points), \n(iii) the sequence of sub-questions should be as simple as possible and only contains the absolutely necessary number of steps to get to the answer of the complex multi-hop question, i.e. each sub-question and its answer must correspond to only one atomic fact and there should be as few sub-questions as possible (2 points), \n(iv) each sub-question in the sequence does not logically contradict the sub-question(s) that comes before it (2 points), and \n(v) as a whole, the sub-questions cover all the necessary information and steps that will allow us to arrive at the same answer as the original complex question (3 points).\n"
                
                else: 
                    prefix_prompt = f"I will provide you with {num} attempts that were made at decomposing a complex multi-hop question into a sequence of simpler sub-questions. Each attempt is indicated by a numerical identifier [] in front of it. Rank the attempts based on their quality with respect to the complex multi-hop question. You should rank the attempts based on the following criteria: \n(i) the sub-questions are grammatically sound, \n(ii) the sub-questions are not ambiguously phrased or overly general such that it becomes difficult to find their answers, \n(iii) the sequence of sub-questions should be as simple as possible and only contains the absolutely necessary number of steps to get to the answer of the complex multi-hop question, i.e. each sub-question and its answer must correspond to only one atomic fact and there should be as few sub-questions as possible, \n(iv) each sub-question in the sequence does not logically contradict the sub-question(s) that comes before it, and \n(v) as a whole, the sub-questions cover all the necessary information and steps that will allow us to arrive at the same answer as the original complex question.\n"
            
            else: raise NotImplementedError
        
        else: 
            prefix_prompt = f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

        return prefix_prompt
    
        ### CHANGE END ###

    def _add_post_prompt(self, query: str, num: int, stance: str = None) -> str:
        ### CHANGE START ### example_ordering and self.rerank_task
        example_ordering = "[02] > [01]" if self._variable_passages else "[04] > [02]"

        if self.rerank_task_name in self.rerank_dqg_tasks:
            # NOTE: 
            # the CQ is added outside; see the following:
            # input_context_cands_post = self.input_context_cands_post.format(query)
            return f"\nRank the {num} attempts above based on their quality with respect to the complex multi-hop question. All the attempts should be included and listed using identifiers, in descending order of quality. If there are several attempts that are equally good amongst themselves, order their ranking by their index number (smallest first). The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."
        
        else: 
            return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."
        ### CHANGE END ###

    def _add_few_shot_examples(self, conv, rank_start = None, rank_end = None):
        assert len(self._examples) >= self._num_few_shot_examples

        for fse_idx in range(self._num_few_shot_examples):
            ### CHANGE START ###
            if self.rerank_task_name in self.rerank_dqg_tasks:
                np.random.seed(54506 + fse_idx)
                ex          = self._examples[fse_idx]
                perm_order  = np.random.permutation(len(ex['paragraphs']))
                num_sys     = rank_end - rank_start
                assert len(perm_order) >= num_sys
                perm_order  = perm_order[:num_sys]
                
                prompt = self.input_context_cands_post.format(ex['query'])

                if fse_idx == 0:
                    # add prefix to front 
                    prefix = self._add_prefix_prompt(None, num_sys)
                    prompt = f"{prefix}\n" + prompt
                
                for rank, pos in enumerate(perm_order):
                    sqs_list    = ex['paragraphs'][pos]
                    if 'breakhigh' in self.rerank_task_name:
                        ops_list = ex['original_lines'][pos]['operators']
                        if type(ops_list) == str: ops_list = eval(ops_list)
                        attempt = ' '.join([f"{self.qpos_tokens[i]} {ops_list[i]} {q}" for i, q in enumerate(sqs_list)])
                    else: 
                        attempt = ' '.join([f"{self.qpos_tokens[i]} {q}" for i,q in enumerate(sqs_list)])

                    prompt      += f"[{str(rank+1).zfill(2)}] {self._replace_number(attempt)}\n"

                order       = np.argsort([ex['scores'][pos] for pos in perm_order])[::-1]
                prompt      += self._add_post_prompt(None, len(perm_order))
                response    = ' > '.join([f'[{i+1}]' for i in order]) + '\n'

                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], response)
            ### CHANGE END ###
            else: 
                ex = random.choice(self._examples)
                obj = json.loads(ex)
                prompt = obj["conversations"][0]["value"]
                response = obj["conversations"][1]["value"]
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], response)
        return conv

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[str, int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        max_length = 300 * (20 / (rank_end - rank_start))
        while True:
            ### CHANGE START ###
            if self.bypass_fsc_load: 
                # force use of rankLLM tempalate, else fastchat will retrieve a very unsuitable default
                conv = get_conversation_template('castorini/rank_zephyr_7b_v1_full')
            else: conv = get_conversation_template(self._model)
            ### CHANGE END ###
            if self._system_message:
                conv.set_system_message(self._system_message)
            ### CHANGE START ### 
            if self._num_few_shot_examples:
                conv = self._add_few_shot_examples(conv, rank_start = rank_start, rank_end = rank_end)
                input_context = ""
            else:
                prefix = self._add_prefix_prompt(query, num)
                input_context = f"{prefix}\n"
            ### CHANGE END ###

            rank = 0
            ### CHANGE START ### zfill(2) & input_context_cands_post
            if self.rerank_task_name in self.rerank_dqg_tasks:
                # input_context_cands_post = f"Complex Multi-hop Query: '{query}'.\n\nPassages: \n"
                input_context_cands_post = self.input_context_cands_post.format(query)
            else: input_context_cands_post = ''
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = self.covert_doc_to_prompt_content(cand.doc, max_length)
                input_context_cands_post += f"[{str(rank).zfill(2)}] {self._replace_number(content)}\n"
                
            input_context_cands_post += self._add_post_prompt(query, num)
            ### CHANGE END ###

            # assume num is fixed (k_candidates is fixed, and window is set.)
            # append the entire prompt (i.e. with the doc cands and post prompt)
            conv.append_message(conv.roles[0], input_context + input_context_cands_post)
            conv.append_message(conv.roles[1], None)
            
            ### CHANGE START ###
            if self.bypass_fsc_load: 
                if self.rerank_task_name in self.rerank_dqg_tasks:
                    messages = [{'role': 'system', 'content': 'You are an intelligent assistant that can rank the quality of a set of decompositions (simpler sub-questions) for a given complex multi-hop question.'}]
                
                else: 
                    messages = [{'role': 'system', 'content': 'You are an intelligent assistant that can rank the relevance of a set of passages to a given query.'}]
                
                c1 = 'mistralai' in self.model_path or 'gemma' in self.model_path
                if c1:
                    # system msg giving jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/...
                    messages = []
                for m in conv.messages:
                    role = m[0].lower().replace('|', '').replace('<', '').replace('>', '')
                    if  role  in ['human', 'user']: message = {'role' : 'user',}
                    elif role in ['assistant']:     message = {'role': 'assistant',}
                    elif role in ['system']:        message = {'role': 'system',}
                    else: raise NotImplementedError(f'🚨\t\tUnknown role: {role}')
                    
                    # apply_chat_template will add the gen prompt (add_generation_prompt = True)
                    if m[1] is None: continue
                    
                    message['content'] = m[1]
                    messages.append(message)

                if 'nvidia/Llama3-ChatQA-1.5-8B' in self.model_path:
                    prompt = get_nvidia_llama3_formatted_input(messages = messages, context = '',
                                                                add_generation_prompt = True)
                else: 
                    prompt = self._tokenizer.apply_chat_template(messages, tokenize = False, 
                                                                  add_generation_prompt = True)   

            else: prompt = conv.get_prompt()
            ### CHANGE END ###
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(
                rank_end - rank_start
            ):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens
                        - self.max_tokens()
                        + self.num_output_tokens(rank_end - rank_start)
                    )
                    // ((rank_end - rank_start) * 4),
                )

        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

### CHANGE START ###
def get_nvidia_llama3_formatted_input(messages, context, add_generation_prompt = False):
    # see https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) 
    ### CHANGE START ###
    if add_generation_prompt: conversation += "\n\nAssistant:"
    ### CHANGE END ###
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input
### CHANGE END ###

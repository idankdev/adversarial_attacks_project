import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, "alignment_attribution"))
from alignment_attribution.lib.eval import eval_zero_shot, not_matched, apply_prompt_template
# from vllm import LLM
# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
# from alignment_attribution.lib.eval import eval_attack
# sys.path.append(os.path.join(ROOT_DIR, "llama2_finetuning_safety/safety_evaluation"))
# sys.path.append(os.path.join(ROOT_DIR, "llama2_finetuning_safety/safety_evaluation/eval_utils"))
# from llama2_finetuning_safety.safety_evaluation.question_inference_caller import run_question_inference
# from llama2_finetuning_safety.safety_evaluation.keyword_eval import main as run_keyword_eval
sys.path.append(os.path.join(ROOT_DIR, "MergeLM"))
from MergeLM.model_merging_methods.mask_weights_utils import mask_model_weights, TaskVector
from decouple import config
from huggingface_hub import login
huggingfacehub_api_token = config("HUGGINGFACE_API_TOKEN")
login(token=huggingfacehub_api_token)
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import json
from copy import deepcopy
from gc import collect
from torch import cuda, no_grad, cuda, randint, bernoulli
from torch import div as torch_div
from torch import load as torch_load
from torch import where as torch_where
from transformers import set_seed
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt

def eval_utility(model_name, model, tokenizer):
    zero_shot_score = eval_zero_shot(model_name, model, tokenizer, limit=200)
    return zero_shot_score

def eval_safety(model, tokenizer, prompt_style="alpaca", output_file=None, num_samples=None):
    batchsize = 24
    advbench_path = "alignment_attribution/data/advbench.txt"
    with open(advbench_path) as f:
        lines = f.readlines()[:num_samples] if num_samples else f.readlines()
    lines = [l.strip("\n").strip() for l in lines]  # remove \n and trailing spaces
    model.resize_token_embeddings(len(tokenizer))
    dialogs = apply_prompt_template(prompt_template_style=prompt_style, dataset=lines)
    batches = [dialogs[i:i + batchsize] for i in range(0, len(dialogs), batchsize)]
    generated_outputs = []
    
    with no_grad():
        for batch in tqdm(batches):
            tokenized_dialogs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256).to("cuda")

            generated_ids = model.generate(**tokenized_dialogs,
                                            max_new_tokens=256,
                                            do_sample=True,
                                            top_p=0,
                                            temperature=1.0,
                                            use_cache=True,
                                            top_k=50,
                                            repetition_penalty=1.0,
                                            length_penalty=1
                                            )
            
            generated_outputs.extend(tokenizer.batch_decode(generated_ids[:, tokenized_dialogs["input_ids"].shape[1] + 1:], skip_special_tokens=True))
            tokenized_dialogs = None
            generated_ids = None
            cuda.empty_cache()
            collect()
            cuda.empty_cache()

    output_scores = np.asarray([not_matched(generated_output) for generated_output in generated_outputs], dtype=np.float32)
    out = [{'prompt': lines[i], 'answer': generated_outputs[i],
            'question_id': str(i), 'ASR_substring_match': str(int(output_scores[i]))} for i in range(len(lines))]

    final_score = output_scores.reshape(-1, 1).max(axis=1).mean()
    if output_file:
        assert (
            output_file != ""
        ), "Please provide a filename to save the attack results."
        with open(output_file, "w") as f:
            for li in out:
                f.write(json.dumps(li))
                f.write("\n")
    return final_score


def get_pretrained_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return model, tokenizer

def get_fine_tuned_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
    
def get_dare_model(pretrained_model, fine_tuned_model, mask_rate):
    param_dict = mask_model_weights(fine_tuned_model,
                            pretrained_model,
                            exclude_param_names_regex=[],
                            weight_format="delta_weight",
                            weight_mask_rate=mask_rate,
                            use_weight_rescale=True,
                            mask_strategy="random")
    dare_model = deepcopy(pretrained_model)
    dare_model.load_state_dict(param_dict)
    dare_model.eval()
    dare_model.requires_grad_(False)
    return dare_model

def eval_safety_statistical(model_name, model, tokenizer):
    # evaluate safety on 15 random seeds
    seeds = randint(0, 100000, (15,))
    safety_scores = []
    for seed in seeds:
        set_seed(seed)
        ASR = eval_safety(model, tokenizer, output_file=None)
        safety_scores.append(ASR)
        print(f"Seed: {seed}, ASR: {ASR}")
    safety_scores = np.array(safety_scores)
    df = pd.DataFrame({"ASR": safety_scores})
    # compute statistics: mean, std, min, max, 25th, 50th, 75th percentiles
    stats = df.describe()
    # save the statistics to a csv file
    stats.to_csv(f"data/attack_results/{model_name}_stats.csv", mode="w", index=True, header=True)
    # save boxplot of safety scores
    df.boxplot(column="ASR")
    plt.ylabel("Safety Scores")
    plt.savefig(f"data/attack_results/{model_name}_boxplot.png")


def get_sparse_task_vector_non_safety_neurons(pretrained_model, finetuned_model, mask_rate):
    filepath = "alignment_attribution/out/llama2-7b-chat-fp16/unstructured/wandg_set_difference_weightonly/FT_mask/mask_bottom_0.500.pt"
    non_safety_neurons = torch_load(filepath)
    task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=[])
    model_param_dict = task_vector.task_vector_param_dict
    masked_param_dict = {}
    with no_grad():
        for param_name, param_value in tqdm(model_param_dict.items()):
            mask = 1 - bernoulli(torch_where(non_safety_neurons[param_name[:-7]], mask_rate, 0))
            masked_param_dict[param_name] = param_value * mask
            if mask_rate != 1.0:
                masked_param_dict[param_name] = torch_div(masked_param_dict[param_name], 1 - mask_rate)
    new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
    masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)
    sparse_model = deepcopy(pretrained_model)
    sparse_model.load_state_dict(masked_param_dict)
    sparse_model.eval()
    sparse_model.requires_grad_(False)
    return sparse_model

from expr_utils import get_pretrained_model, get_fine_tuned_model, get_dare_model, eval_safety, eval_utility, eval_safety_statistical
from decouple import config
from huggingface_hub import login
huggingfacehub_api_token = config("HUGGINGFACE_API_TOKEN")
login(token=huggingfacehub_api_token)
from torch import linspace
from tqdm import tqdm
from torch import cuda, random, randint, cuda, cat
from gc import collect
import pandas as pd
import numpy as np


SEED = 0
random.manual_seed(SEED)
device = "cuda" if cuda.is_available() else "cpu"
LOG_FILE = f"data/attack_results/dare_results.csv"
model_name = "TheBloke/Llama-2-7B-Chat-fp16"
pretrained_model_save_path = "llama2_finetuning_safety/ckpts/Llama-2-7b-chat-fp16"
fine_tuned_model_save_path = "llama2_finetuning_safety/finetuned_models/alpaca-7b-full"

if __name__ == "__main__":
    pretrained_model, tokenizer = get_pretrained_model("llama2_finetuning_safety/ckpts/Llama-2-7b-chat-fp16")
    fine_tuned_model = get_fine_tuned_model("llama2_finetuning_safety/finetuned_models/alpaca-7b-full")
    mask_rates = cat([linspace(0, 1, 11)[:-1], linspace(.9, 1, 10)], dim=0)
    results = []
    for i, mask_rate in tqdm(enumerate(mask_rates)):
        mask_rate = round(mask_rate.item(), 3)
        ASRs = []
        for _ in range(10):
            seed = randint(0, 1000, (1,)).item()
            random.manual_seed(seed)
            dare_model = get_dare_model(pretrained_model, fine_tuned_model, mask_rate=mask_rate)
            dare_model.to(device)
            ASRs.append(eval_safety(dare_model, tokenizer, "alpaca", num_samples=100))
            # metrics["base_ASR"] = eval_safety(dare_model, tokenizer, "base", output_file="data/test_base.json")
            # metrics["pure_bad_ASR"] = eval_safety(dare_model, tokenizer, "pure_bad", output_file="data/test_pure_bad.json")
            # metrics["aoa_ASR"] = eval_safety(dare_model, tokenizer, "aoa", output_file="data/test_aoa.json")
            # utility_metrics = eval_utility(model_name, dare_model, tokenizer)["results"]
            # metrics["utility_acc_avg"] = sum([val["acc"] for val in utility_metrics.values()]) / len(utility_metrics)
            # utility_metrics = pd.json_normalize(utility_metrics).to_dict(orient="records")[0]
            # metrics.update(utility_metrics)
            dare_model.to("cpu")
            dare_model = None
            cuda.empty_cache()
            collect()
            cuda.empty_cache()
            print(cuda.memory_summary())
        metrics = {}
        metrics["mask_rate"] = mask_rate
        metrics["alpaca_ASR"] = np.mean(ASRs)
        metrics["alpaca_ASR_std"] = np.std(ASRs)
        print(metrics)
        results.append(metrics)
    df = pd.DataFrame(results, index=None)
    df.to_csv(LOG_FILE, mode="w", index=False, header=True)

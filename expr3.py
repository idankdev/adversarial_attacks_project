from expr_utils import get_pretrained_model, get_fine_tuned_model, get_sparse_task_vector_non_safety_neurons, eval_safety
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
LOG_FILE = f"data/attack_results/sparsed_results.csv"
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
            sparsed_model = get_sparse_task_vector_non_safety_neurons(pretrained_model, fine_tuned_model, mask_rate)
            sparsed_model.to(device)
            ASRs.append(eval_safety(sparsed_model, tokenizer, "alpaca", output_file=None, num_samples=100))
            sparsed_model.to("cpu")
            sparsed_model = None
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

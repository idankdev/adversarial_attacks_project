#!/bin/bash

JOB_NAME="safety_eval"
CONDA_ENV=prune_llm
CONDA_HOME=$HOME/miniconda3
NUM_CORES=2
NUM_NODES=1
NUM_GPUS=1
GPU_TYPE=L40
PRETRAINED_MODEL_PATH="TheBloke/Llama-2-7B-Chat-fp16"

sbatch \
	-A cs \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$GPU_TYPE:$NUM_GPUS \
	--job-name $JOB_NAME \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

gpustat

python -u safety_evaluation/question_inference.py \
--model_name finetuned_models/alpaca-7b-full \
--prompt_file safety_evaluation/data/harmful_behaviors.csv \
--prompt_template_style alpaca \
--output_file safety_evaluation/question_output/harmful_behaviors_alpaca_7b_full.jsonl && \
python safety_evaluation/keyword_eval.py --input_file safety_evaluation/question_output/harmful_behaviors_alpaca_7b_full.jsonl
     
     

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

#!/bin/bash

JOB_NAME="expr3"
CONDA_ENV=prune_llm
CONDA_HOME=$HOME/miniconda3
NUM_CORES=1
NUM_NODES=1
NUM_GPUS=1
GPU_TYPE=L40

model="llama2-7b-chat-fp16"
method="wandg_set_difference" #"wandg"
type="unstructured"
suffix="weightonly"
dataset="align"
save_dir="out/$model/$type/${method}_${suffix}" #"out/$model/$type/${method}_${suffix}/$dataset"

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

python main.py \
    --model $model \
    --prune_method $method \
    --sparsity_ratio 0.5 \
    --prune_data $dataset \
    --p 0.1\
    --q 0.1\
    --sparsity_type $type \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res \
    --save_mask \
    --seed 42 #\
    # --neg_prune \
    # --dump_wanda_score
     
echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

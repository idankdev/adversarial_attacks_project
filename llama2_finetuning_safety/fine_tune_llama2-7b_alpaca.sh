#!/bin/bash

JOB_NAME="fine_tune"
CONDA_ENV=prune_llm
CONDA_HOME=$HOME/miniconda3
NUM_CORES=4
NUM_NODES=1
NUM_GPUS=2
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

torchrun --nnodes $NUM_NODES --nproc_per_node $NUM_GPUS finetuning.py \
--batch_size_training 32 --lr 2e-5 \
--gradient_accumulation_steps 1 --weight_decay 0 \
--num_epochs 1 \
--dataset alpaca_dataset \
--enable_fsdp \
--model_name $PRETRAINED_MODEL_PATH --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder alpaca-7b-full && \
python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/alpaca-7b-full-epoch=1-TheBloke/Llama-2-7B-Chat-fp16" -consolidated_model_path "finetuned_models/alpaca-7b-full/" -HF_model_path_or_name $PRETRAINED_MODEL_PATH
     

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

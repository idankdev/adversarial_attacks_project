#!/bin/bash

JOB_NAME="expr3"
CONDA_ENV=prune_llm
CONDA_HOME=$HOME/miniconda3
NUM_CORES=1
NUM_NODES=1
NUM_GPUS=1
GPU_TYPE=L40

sbatch \
	-A cs \
	-w tdk-bm4 \
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

python ./expr3.py
     
echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

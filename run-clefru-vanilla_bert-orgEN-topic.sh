#!/bin/bash
#$ -q gpu.q -cwd
#$ -l h_rt=8:00:00,gpu=1
#$ -t 1:5
#$ -N clefru-vanilla_bert-orgEN-topic
#$ -j y

module load cuda11.0/toolkit/11.0.3
module load cuda11.0/blas/11.0.3

i=${SGE_TASK_ID}
conda activate opennir
scripts/pipeline.sh config/vanilla_bert config/clef0304/orgEN-topic config/clef0304/folds/f$i
echo "===== config/vanilla_bert config/clef0304/orgEN-topic config/clef0304/folds/f$i ===="

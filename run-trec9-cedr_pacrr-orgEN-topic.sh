#!/bin/bash
#$ -q gpu.q -cwd
#$ -l h_rt=8:00:00,gpu=1
#$ -t 5
#$ -N trec9-cedr_pacrr-orgEN-topic
#$ -j y

module load cuda11.0/toolkit/11.0.3
module load cuda11.0/blas/11.0.3

i=${SGE_TASK_ID}
conda activate opennir
scripts/pipeline.sh config/cedr/pacrr config/trec9/orgEN-topic config/trec9/folds/f$i \
                    train_ds.docversion=mtzhen20210121.scale18ts1-ersatz \
                    valid_ds.docversion=mtzhen20210121.scale18ts1-ersatz \
                    test_ds.docversion=mtzhen20210121.scale18ts1-ersatz
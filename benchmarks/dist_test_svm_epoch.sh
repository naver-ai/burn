#!/bin/bash
set -e
set -x

CFG=$1
CKPT=$2
FEAT_LIST=$3 # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
GPUS=$4
WORK_DIR=$5

mkdir -p $WORK_DIR/logs
echo "Testing checkpoint: $CKPT" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/dist_extract.sh $CFG $GPUS $WORK_DIR --checkpoint $CKPT

bash benchmarks/svm_tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

bash benchmarks/svm_tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"

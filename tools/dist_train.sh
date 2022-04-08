#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1 # type of configuration
GPUS=$2 # number of gpus
PY_ARGS=${@:3}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${PY_ARGS}

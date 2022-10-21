#!/bin/bash
WS=`pwd`
CUDA_DEVICE_ORDER=PCI_BUS_ID
LOCAL_RUN="xargs -L1 python"

n_types=25
dataset="MemeTracker-0.0M-$n_types"
shared_args="--dataset $dataset --skip_pred_next_event --verbose"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi

if [[ $* == *all* ]] || [[ $* == *ISAHP* ]]; then
    printf "%s\n" "$WS/tasks/train_isahp.py ISAHP $shared_args --type_reg 5. --l1_reg 0. --lr 0.001 --epochs 50 --batch_size 16 --embedding_dim 49 --hidden_size 50 --num_head 2 --cuda --split_id "{0..0} | $LOCAL_RUN
fi

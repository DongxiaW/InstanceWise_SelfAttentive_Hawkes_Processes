#!/bin/bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
WS=`pwd`

n_seqs=1000
n_types=10
dataset=pgem-$(($n_seqs / 1000))K-$n_types
shared_args="--dataset $dataset"
n_splits=5  # if modified, remember to modify below as well!!!

LOCAL_RUN="xargs -L1 python"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi

if [[ $* == *all* ]] || [[ $* == *preprocess* ]]; then
    python preprocessing/generate_events_by_pgem.py \
        --n_seqs $n_seqs \
        --n_copies 2 \
        --max_t 500
fi

# training for each methods

if [[ $* == *all* ]] || [[ $* == *Tran* ]]; then
    printf "%s\n" "$WS/tasks/train.py Tran $shared_args --max_mean 20 --n_bases 7 --cuda --split_id "{0..4} | $LOCAL_RUN
fi

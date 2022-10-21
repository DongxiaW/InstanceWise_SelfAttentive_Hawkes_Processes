#!/bin/bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
WS=`pwd`

n_seqs=1000
n_types=5
dataset=pgem-$(($n_seqs / 1000))K-$n_types
shared_args="--dataset $dataset"
n_splits=5

LOCAL_RUN="xargs -L1 python"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi

if [[ $* == *preprocess* ]]; then
    python preprocessing/generate_events_by_pgem.py \
        --n_seqs $n_seqs \
        --n_copies 1 \
        --max_t 30
fi


if [[ $* == *all* ]] || [[ $* == *ISAHP* ]]; then
    printf "%s\n" "$WS/tasks/train_isahp.py ISAHP $shared_args --type_reg 0.25 --l1_reg 0.025 --lr 0.001 --epochs 200 --batch_size 8 --embedding_dim 9 --hidden_size 10 --num_head 2 --cuda --split_id "{0..0} | $LOCAL_RUN
fi



#!/bin/bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
WS=`pwd`

n_seqs=1000
n_types=15
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
        --n_copies 3 \
        --max_t 1000
fi


if [[ $* == *all* ]] || [[ $* == *Tran* ]]; then
    printf "%s\n" "$WS/tasks/train_tran.py Tran $shared_args --tran_layer 1 --tran_head 3 --max_mean 20 --n_bases 7 --cuda --split_id "{0..4} | $LOCAL_RUN
    printf "%s\n" "$WS/tasks/train_tran.py Tran $shared_args --tran_layer 2 --tran_head 3 --max_mean 20 --n_bases 7 --cuda --split_id "{0..4} | $LOCAL_RUN
    printf "%s\n" "$WS/tasks/train_tran.py Tran $shared_args --tran_layer 3 --tran_head 3 --max_mean 20 --n_bases 7 --cuda --split_id "{0..4} | $LOCAL_RUN
fi

# python postprocessing/summarize_results.py $dataset

# Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes


## Paper: 
Dongxia Wu, Tsuyoshi Idé, Aurélie Lozano, Georgios Kollias, Jiří Navrátil, Naoki Abe, Yi-An Ma, Rose Yu, [Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes](https://arxiv.org/abs/2402.03726), AISTATS 2024

## Requirements

Dependency can be installed using the following command:
```bash
conda env create -n <myenv> -f environment.yml
conda activate <myenv>
```

## Abstract
We address the problem of learning Granger causality from asynchronous, interdependent, multi-type event sequences. In particular, we are interested in discovering instance-level causal structures in an unsupervised manner. Instance-level causality identifies causal relationships among individual events, providing more fine-grained information for decision-making. Existing work in the literature either requires strong assumptions, such as linearity in the intensity function, or heuristically defined model parameters that do not necessarily meet the requirements of Granger causality. We propose Instance-wise Self-Attentive Hawkes Processes (ISAHP), a novel deep learning framework that can directly infer the Granger causality at the event instance level. ISAHP is the first neural point process model that meets the requirements of Granger causality. It leverages the self-attention mechanism of the transformer to align with the principles of Granger causality. We empirically demonstrate that ISAHP is capable of discovering complex instance-level causal structures that cannot be handled by classical models. We also show that ISAHP achieves state-of-the-art performance in proxy tasks involving type-level causal discovery and instance-level event type prediction.


## How to Run

Run the scripts for synergy and MT datasets
```
./scripts/run_synergy.sh ISAHP
./scripts/run_mt.sh ISHAP
```

## Cite
```
@inproceedings{wu2024learning,
  title={Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes},
  author={Wu, Dongxia and Id{\'e}, Tsuyoshi and Lozano, Aur{\'e}lie and Kollias, Georgios and Navr{\'a}til, Ji{\v{r}}{\'\i} and Abe, Naoki and Ma, Yi-An and Yu, Rose},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence and Statistics},
  year={2024}
}
```

# On Exposure Bias, Hallucination and Domain Shift in Neural Machine Translation

This repository is the training scripts of [On Exposure Bias, Hallucination and Domain Shift in Neural Machine Translation](https://arxiv.org/abs/2005.03642). The code has been merged into [Nematus](https://github.com/EdinburghNLP/nematus). The initial version of codes for the paper is in branch [`initial-version`](https://github.com/zippotju/Exposure-Bias-Hallucination-Domain-Shift/tree/initial-version).

## Usage instruction
*Only respository-specific usage instruction, for general usages instruction, please refer to [Nematus](https://github.com/EdinburghNLP/nematus)*

#### `nematus/train.py` : use to train a new model

#### training parameters
| parameter | description |
|---        |---          |
| --loss_function {cross-entropy,per-token-cross-entropy, MRT} | loss function. MRT: Minimum Risk Training https://www.aclweb.org/anthology/P/P16/P16-1159.pdf) (default: cross-entropy) |
| --print_per_token_pro PATH | PATH to store the probability of each target token given source sentences over the training dataset (without training). If set to False, the function will not be triggered. (default: False). Please get rid of the 1.0s at the end of each list which are the probability of padding. |

#### minimum risk training parameters (MRT)

| parameter | description |
|---        |---          |
| --mrt_reference | add reference into MRT candidates sentences (default: False) |
| --mrt_alpha FLOAT | MRT alpha to control the sharpness of the distribution of sampled subspace (default: 0.005) |
| --samplesN INT | the number of sampled candidates sentences per source sentence (default: 100) |
| --mrt_loss | evaluation metrics used to compute loss between the candidate translation and reference translation (default: SENTENCEBLEU n=4) |
| --mrt_ml_mix FLOAT | mix in MLE objective in MRT training with this scaling factor (default: 0) |
| --sample_way {beam_search, randomly_sample} | the sampling strategy to generate candidates sentences (default: beam_search) |
| --max_len_a INT | generate candidates sentences with maximum length: ax + b, where x is the length of the source sentence (default: 1.5) |
| --max_len_b INT | generate candidates sentences with maximum length: ax + b, where x is the length of the source sentence (default: 5) |
| --max_sentences_of_sampling INT | maximum number of source sentences to generate candidates sentences at one time (limited by device memory capacity) (default: 0) |

Data dowload and preprocessing, please refer to https://github.com/ZurichNLP/domain-robustness

Please refer to https://github.com/EdinburghNLP/wmt17-transformer-scripts for training and evaluation process, the default training scripts of our experiments are [here](./scripts).



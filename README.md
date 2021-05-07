# Game-based Neural Architecture Search

This is the code for the paper "Game-based Neural Architecture Search"

# Requirements

Python >= 3.6.11

Pytorch >= 1.6.0

# Architecture Search

To search CNN architectures by GameNAS and the improved random search, run
```
bash pytorch_GameNAS/train_search.sh
```
Additionally, we provide the random seeds in ``` train_search.sh ``` to reproduce the distribution figures in our paper.

To search CNN architectures with regularization, run
```
bash pytorch_GameNAS/train_search_regularized.sh
```


# Architecture Evaluation

After searching, the architectures returned by GameNAS are given in ```GameNAS_Arcs/```, along with their validation accuracies. We choose the architectures with high validation accuracies and train them further.

To train a found architecture in CIFAR-10, run
```
bash pytorch_GameNAS/train_final.sh
```
The default architecture is an architecture with 2.45% test error.

To transfer a found architecture to ImageNet, run
```
bash pytorch_GameNAS/train_final_imagenet.sh
```
The default architecture is an architecture with 25.78% top-1 error.

# Acknowledgements
We thank Liu et al. for the discussion on some training details in [`DARTS`](https://github.com/quark0/darts) implementation. 
We furthermore thank the anonymous reviewers for their constructive comments.


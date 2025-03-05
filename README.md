# Severing Spurious Correlations with Data Pruning

This repository contains the code for experiments in the ICLR 2025 Spotlight paper [Severing Spurious Correlations with Data Pruning](https://openreview.net/forum?id=Bk13Qfu8Ru) by Varun Mulchandani and Jung-Eun Kim.

[03/04/2025] More experiments to come soon!

## CelebA Experiments

Please download and unzip the CelebA dataset from [Liu et. al. 2015](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (Aligned and Cropped Images) and move it to the CelebA/data/ sub-directory.

Sample Commands

```
# Create train, test and val splits.

bash create_init_dataset.sh

# Train model on unpruned dataset.

python3 generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 25 --lr 1e-3 --weight_decay 1e-4 --method ERM --prune False
bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_25_lr_0.001_weight_decay_0.0001/job.sh

# Save the original metadata with a different name.

bash change_meta.sh

# Prune the dataset. prune_oracle and prune_general take amount to be pruned between 0.1 - 0.97. Type of pruning can be "hardest" or "easiest".

python3 prune_oracle.py 0.5 hardest # Assuming access to spurious information. Reproduces results in Fig. 4.
python3 prune_general.py 0.5 hardest # Assuming access to no information. Reproduces results in Fig. 6. 
bash prune_metadata.sh

# Train on pruned dataset. Make sure to move the original training results elsewhere, or they'll get overwritten.

python3 generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 25 --lr 1e-3 --weight_decay 1e-4 --method ERM --prune True
bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_25_lr_0.001_weight_decay_0.0001/job.sh

```

Please consider citing our paper if you find this repository useful in your work.

```bibtex
@inproceedings{
mulchandani2025severing,
title={Severing Spurious Correlations with Data Pruning},
author={Varun Mulchandani and Jung-Eun Kim},
booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
year={2025},
url={https://openreview.net/forum?id=Bk13Qfu8Ru}
}
```

## References

Many of our experiments are built on top of implementations provided by the following papers:

1. [Just Train Twice: Improving Group Robustness without Training Group Information](https://arxiv.org/pdf/2107.09044)

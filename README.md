# Instructions

## CelebA Experiments

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


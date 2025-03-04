import pandas as pd

# An example path 'results/CelebA/CelebA_sample_exp/rel2n/ERM_upweight_0_epochs_25_lr_0.001_weight_decay_0.0001/model_outputs/'
d1 = pd.read_csv(path + 'test.csv')

print('Present final results: No tuning.')

print('Worst Group Accuracy: ', d1.loc[d1.shape[0] - 1, 'avg_acc_group:1'])
print('Test Accuracy: ', d1.loc[d1.shape[0] - 1, 'avg_acc'])

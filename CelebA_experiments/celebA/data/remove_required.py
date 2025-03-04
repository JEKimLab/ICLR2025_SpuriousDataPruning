import torch
import pandas as pd
import numpy as np
import os
import random

random.seed(1)

d1 = pd.read_csv('original_metadata.csv')
d2 = pd.read_csv('original_list_eval_partition.csv')

to_include = []
count1 = 0
count2 = 0
count3 = 0

for i in range(d1.shape[0]):
    if d2.loc[i, 'split'] != 0:
        to_include.append(i)
    elif d1.loc[i, 'Male'] == 1 and d1.loc[i, 'Eyeglasses'] == -1 and count1 < 2500:
        to_include.append(i)
        count1 += 1
    elif d1.loc[i, 'Male'] == 1 and d1.loc[i, 'Eyeglasses'] == 1:
        if count2 < 2500:
            to_include.append(i)
        count2 += 1
    elif d1.loc[i, 'Male'] == -1 and d1.loc[i, 'Eyeglasses'] == -1 and count3 < 5000:
        to_include.append(i)
        count3 += 1
    elif d1.loc[i, 'Male'] == -1 and d1.loc[i, 'Eyeglasses'] == 1:
        to_include.append(i)
        if random.random() > 0.5:
            d2.loc[i, 'split'] = 1
        else:
            d2.loc[i, 'split'] = 2

d1_final = d1.loc[to_include]
d2_final = d2.loc[to_include]

d2_final.to_csv('list_eval_partition.csv')
d1_final.to_csv('metadata.csv')

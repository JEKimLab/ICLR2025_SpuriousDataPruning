import pickle
from collections import Counter
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('prune_amount', type=float, default=0.5)
parser.add_argument('prune_type', type=str, default="hardest")
args = parser.parse_args()

d1 = pd.read_csv('celebA/data/metadata_main.csv')
with open('9el2n.pkl', 'rb') as f:
    d = pickle.load(f)

x = []
for i, j, k in d:
    x.append(i)

if args.prune_type == "hardest":
    d.sort(key = lambda x:-x[1])
else:
    d.sort(key = lambda x:x[1])

curr_req = []
counter_1 = 0
for i, j, k in d:
    if counter_1 < 2500*args.prune_amount:
        if k == 1 and int(d1.loc[i, 'Eyeglasses']) == 1:
            curr_req.append(i)
            counter_1 += 1

with open('to_prune.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

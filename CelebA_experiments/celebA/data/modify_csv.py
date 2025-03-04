import pandas as pd
import pickle

with open('../../to_prune.pkl', 'rb') as f:
    x = pickle.load(f)

m = pd.read_csv('metadata_main.csv')
m_other = pd.read_csv('list_eval_partition_main.csv')

m1_new = m[~m.index.isin(x)]
m1_new.to_csv('metadata.csv')

m1_new = m_other[~m_other.index.isin(x)]
m1_new.to_csv('list_eval_partition.csv')

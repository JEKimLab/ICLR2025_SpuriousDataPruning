import pickle
import os
from collections import defaultdict

with open('10el2n.pkl', 'rb') as f:
    x = pickle.load(f)

third_1 = []
third_2 = []
third_3 = []

x.sort(key = lambda x:x[2])

counter = 0

for i in x:
    if i[1] == 0:
        if counter < 100:
            third_1.append(i[0])
        elif counter >= 2900 and counter < 3000:
            third_2.append(i[0])
        elif counter >= 4900 and counter < 5000:
            third_3.append(i[0])
        counter += 1

with open('all_thirds.pkl', 'wb') as f:
    pickle.dump([third_1, third_2, third_3], f)

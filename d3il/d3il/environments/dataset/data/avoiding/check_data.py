import os

import numpy as np

data_dir = "data/"

data_list = os.listdir(data_dir)

lengths = []
for file in data_list:

    arr = np.load(data_dir + file, allow_pickle=True)

    lengths.append(len(arr["robot"]["c_pos"]))

lengths = np.array(lengths)

print("data points: ", np.sum(lengths))

print("mean: ", np.mean(lengths))
print("std: ", np.std(lengths))
print("max: ", np.max(lengths))
print("min: ", np.min(lengths))

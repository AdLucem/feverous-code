import json
import pickle
import numpy as np

dirname = "/home/kw/Data/"

# with open(dirname + 'dev_data_run.json', "w+") as f:
#    dev_data = json.load(f)

with open(dirname + "logits.npy", "wb+") as f:
    logits = np.load(f)

print(logits.shape)
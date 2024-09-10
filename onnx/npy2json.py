import numpy as np
import json


array = np.load("../model_assets/model_assets/tsukuyomi/style_vectors.npy")
data = array.tolist()


with open("style_vectors.json", "w") as f:
    json.dump({
        "data": data,
        "shape": array.shape,
    }, f)
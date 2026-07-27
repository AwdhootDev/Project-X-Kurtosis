import scipy.io
import numpy as np

def load_signal(file_path):
    data = scipy.io.loadmat(file_path)
    signal = None

    for key in data.keys():
        if key.endswith("_DE_time"):
            signal = data[key].flatten()
            break
    
    return signal
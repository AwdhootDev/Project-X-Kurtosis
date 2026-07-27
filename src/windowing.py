import numpy as np

def create_windows(signal):
    window_size = 2048
    windows = []

    for start in range(0, len(signal) - window_size, 2048):
        window = signal[start : start + window_size]
        windows.append(window)
    
    return np.array(windows)
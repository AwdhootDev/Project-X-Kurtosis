import numpy as np

def create_windows(signal):
    window_size = 2048
    step = 1024
    windows_train = []
    windows_test = []

    split_idx = int(len(signal) * 0.7)

    train_signal = signal[:split_idx]
    test_signal = signal[split_idx + window_size:]

    #Overlapping in train data for complexation
    for start in range(0, len(train_signal) - window_size, step):
        window = train_signal[start : start + window_size]
        windows_train.append(window)

    #NonOverlapping in test
    for start in range(0, len(test_signal) - window_size, 2048):
        window = test_signal[start : start + window_size]
        windows_test.append(window)
    
    return np.array(windows_train), np.array(windows_test)
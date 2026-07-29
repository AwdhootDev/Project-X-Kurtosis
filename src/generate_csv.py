import os
import pandas as pd

from sklearn.model_selection import train_test_split
from loader import load_signal
from windowing import create_windows
from time_features import extract_features
from frequency_features import extract_frequency_features

folders = {
    "Normal" : "Data/normal",
    "Ball Fault" : "Data/12k_DE_fault/Ball fault",
    "Inner race" : "Data/12k_DE_fault/Inner race",
    "Outer race" : "Data/12k_DE_fault/Outer race"
}

label_map = {
    "Normal" : 0,
    "Ball Fault" : 1,
    "Inner race" : 2,
    "Outer race" : 3
}

train_rows = []
test_rows = []

for label, folder in folders.items():
    for file in os.listdir(folder):
        if not file.endswith(".mat"):
            continue

        file_path = os.path.join(folder, file)
        signal = load_signal(file_path)
        windows_train, windows_test = create_windows(signal)

        for window in windows_train:
            time_features = extract_features(window)
            freq_features = extract_frequency_features(window)

            features = {**time_features, **freq_features}

            features["label"] = label_map[label]
            features["file_name"] = file
            train_rows.append(features)

        for window in windows_test:
            time_features = extract_features(window)
            freq_features = extract_frequency_features(window)

            features = {**time_features, **freq_features}

            features["label"] = label_map[label]
            features["file_name"] = file
            test_rows.append(features)

df_train = pd.DataFrame(train_rows)
df_test = pd.DataFrame(test_rows)

df_train.to_csv("Output/train_features.csv", index=False)
df_test.to_csv("Output/test_features.csv", index=False)
# print(df.head())
# print("Total Samples", len(df))
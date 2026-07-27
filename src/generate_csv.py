import os
import pandas as pd

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

rows = []

for label, folder in folders.items():
    for file in os.listdir(folder):
        if not file.endswith(".mat"):
            continue

        file_path = os.path.join(folder, file)
        signal = load_signal(file_path)
        windows = create_windows(signal)

        for window in windows:
            time_features = extract_features(window)
            freq_features = extract_frequency_features(window)

            features = {**time_features, **freq_features}

            features["label"] = label
            features["file_name"] = file
            rows.append(features)

df = pd.DataFrame(rows)

df.to_csv("Output/features.csv", index=False)
print(df.head())
print("Total Samples", len(df))
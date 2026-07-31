import os
import glob
import re
import numpy as np
import scipy.io as sio

WND_SIZE = 2048

def parse_filename(filepath):
    """
    Extracts the 10-class label and the motor load (HP).
    Handles both numerical 'normal' files and descriptive 'fault' files.
    """
    filename = os.path.basename(filepath).lower()

    if '97.mat' in filename: return 0, 0   # Label 0, 0 HP
    if '98.mat' in filename: return 0, 1   # Label 0, 1 HP
    if '99.mat' in filename: return 0, 2   # Label 0, 2 HP
    if '100.mat' in filename: return 0, 3  # Label 0, 3 HP

    load_match = re.search(r'_(\d)\.mat$', filename)
    if not load_match:
        raise ValueError(f"Could not find load (HP) in filename: {filename}")
        
    load_hp = int(load_match.group(1))

    if 'b007' in filename: label = 1
    elif 'b014' in filename: label = 2
    elif 'b021' in filename: label = 3
    elif 'ir007' in filename: label = 4
    elif 'ir014' in filename: label = 5
    elif 'ir021' in filename: label = 6
    elif 'or007' in filename: label = 7
    elif 'or014' in filename: label = 8
    elif 'or021' in filename: label = 9
    else:
        raise ValueError(f"Could not determine class for file: {filename}")
        
    return label, load_hp

def extract_data(target_loads, folder_path="data/**/*.mat"):
    """
    Crawls the dataset and extracts windows ONLY if the file's load matches the target_loads list.
    """
    X_data = []
    y_labels = []
    
    for file in glob.glob(folder_path, recursive=True):
        try:
            label, load_hp = parse_filename(file)
        except ValueError as e:
            continue

        if load_hp in target_loads:
            mat_file = sio.loadmat(file)
            mat_key = None

            for key in mat_file.keys():
                if "_DE_time" in key:
                    mat_key = key
                    break
                    
            if mat_key is None:
                continue
                
            signal = mat_file[mat_key].flatten()
            
            no_windows = len(signal) // WND_SIZE
            signal = signal[:no_windows * WND_SIZE]
            x_matrix = signal.reshape(no_windows, WND_SIZE)
            y_matrix = np.array([label] * no_windows)
            
            X_data.append(x_matrix)
            y_labels.append(y_matrix)
            
    return np.vstack(X_data), np.concatenate(y_labels)

print("Starting Domain Generalization Extraction...")

print("Extracting Train Data (0, 1, 2 HP)...")
X_train, y_train = extract_data(target_loads=[0, 1, 2])
np.savez("cwru_train_multi_load.npz", X=X_train, y=y_train)
print(f"Train Set Saved! Shape: {X_train.shape}, Unique Classes: {np.unique(y_train)}\n")

print("Extracting Test Data (3 HP)...")
X_test, y_test = extract_data(target_loads=[3])
np.savez("cwru_test_3HP.npz", X=X_test, y=y_test)
print(f"Test Set Saved! Shape: {X_test.shape}, Unique Classes: {np.unique(y_test)}")
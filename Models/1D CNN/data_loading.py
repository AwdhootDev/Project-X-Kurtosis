import os
import numpy as np
import scipy.io

WND_SIZE = 2048

X_main = []
y_main = []

folder_label_map = {
    "data/normal": 0,
    "data/12k_DE_fault/Ball fault": 1,
    "data/12k_DE_fault/Inner race": 2,
    "data/12k_DE_fault/Outer race": 3
}

for folder_path, label in folder_label_map.items():
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mat"):
            full_path = os.path.join(folder_path, file_name)
            
            my_dict = scipy.io.loadmat(full_path)

            de_key = None
            for key in my_dict.keys():
                if '_DE_time' in key:
                    de_key = key
                    break
            raw_data = my_dict[de_key].flatten()
            no_of_window = len(raw_data)//WND_SIZE

            cutoff = no_of_window*WND_SIZE
            X_matrix = raw_data[ :cutoff]
            X_matrix = X_matrix.reshape(no_of_window,WND_SIZE)
            y_label = np.array([label]* no_of_window)
            X_main.append(X_matrix)
            y_main.append(y_label)

X_final = np.vstack(X_main)
y_final = np.concatenate(y_main)

print("Final X shape:", X_final.shape)
print("Final y shape:", y_final.shape)

save_path = "cwru_dataset.npz"
np.savez(save_path, X=X_final, y=y_final)
print(f"Dataset successfully saved to {save_path}!")

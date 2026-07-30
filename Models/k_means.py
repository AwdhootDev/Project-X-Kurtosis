import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("../Output/features.csv")
x = df.drop(columns=['label', 'file_name'])
y = df['label']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
# print(encoder.classes_) # 0 for ball, 1 for inner, 2 normal, 3 outer

kmeans = KMeans(n_clusters=4,random_state=42, n_init='auto')

cluster = kmeans.fit_predict(x_scaled)
mapped_labels = np.zeros_like(cluster)
for i in range(4):
    rows = (cluster == i)
    actual = y_encoded[rows]

    if len(actual)>0 :
        most_common = np.bincount(actual).argmax()

    mapped_labels[rows] = most_common

acc = accuracy_score(y_encoded, mapped_labels)
prec = precision_score(y_encoded, mapped_labels, average='weighted', zero_division=0)
rec = recall_score(y_encoded, mapped_labels, average='weighted', zero_division=0)
f1 = f1_score(y_encoded, mapped_labels, average='weighted', zero_division=0)

print("\n--- K-Means Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_encoded, mapped_labels)

cm_df = pd.DataFrame(
    cm, 
    index=[f'Actual {i}' for i in range(4)], 
    columns=[f'Predicted {i}' for i in range(4)]
)

print(cm_df.to_string())


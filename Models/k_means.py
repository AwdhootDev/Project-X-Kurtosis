import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("../Output/features.csv")
df['label'] = (df['label'] == 'Normal').astype(int)
df_ = df[df['label']==1]

x = df_.drop(columns=['label', 'file_name'])
y = df_['label']

x_test = df.drop(columns=['label', 'file_name'])
y_test = df['label']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_test = scaler.transform(x_test)

kmeans = KMeans(n_clusters=2,random_state=42, n_init='auto')

cluster = kmeans.fit(x_scaled)
y_pred = cluster.predict(x_test)

mapped_labels = np.empty_like(y_pred, dtype=object)
for i in range(2):
    rows = (y_pred == i)
    actual = y_test.values[rows]
    if len(actual) > 0:
        unique_labels, counts = np.unique(actual, return_counts=True)
        most_common = unique_labels[np.argmax(counts)]
        mapped_labels[rows] = most_common

mapped_labels = mapped_labels.astype(str)
y_test = y_test.astype(str)

acc = accuracy_score(y_test, mapped_labels)
prec = precision_score(y_test, mapped_labels, average='weighted', zero_division=0)
rec = recall_score(y_test, mapped_labels, average='weighted', zero_division=0)
f1 = f1_score(y_test, mapped_labels, average='weighted', zero_division=0)

print("\n--- K-Means Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, mapped_labels)

cm_df = pd.DataFrame(
    cm, 
    index=[f'Actual {i}' for i in range(cm.shape[0])], 
    columns=[f'Predicted {i}' for i in range(cm.shape[1])]
)
print(cm_df.to_string())


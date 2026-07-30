import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

df_train = pd.read_csv("../Output/train_features.csv")
df_test = pd.read_csv("../Output/test_features.csv")

x_train = df_train.drop(columns=['label', 'file_name'])
y_train = df_train['label']

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
# print(encoder.classes_) 0 ball , 1 inner, 2 normal, 3 outer
x_test = df_test.drop(columns=['label', 'file_name'])
y_test = df_test['label']

y_test = encoder.transform(y_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

svm_c = SVC(random_state=42, C= 1)


print("Model training")
svm_c.fit(x_train_scaled,y_train)

y_pred = svm_c.predict(x_test_scaled)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test, y_pred, average = 'weighted')
rec = recall_score(y_test,y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')

print("\n--- SVM Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, 
    index=[f'Actual {c}' for c in svm_c.classes_], 
    columns=[f'Predicted {c}' for c in svm_c.classes_]
)

print(cm_df.to_string())

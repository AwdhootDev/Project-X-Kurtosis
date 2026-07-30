import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, recall_score, precision_score,f1_score

df_train = pd.read_csv("../Output/train_features.csv")
df_test = pd.read_csv("../Output/test_features.csv")

x_train = df_train.drop(columns=['label', 'file_name'])
y_train = df_train['label']

x_test = df_test.drop(columns=['label', 'file_name'])
y_test = df_test['label']

ada = AdaBoostClassifier(n_estimators=650, random_state=42, learning_rate= 1)
print("Training model")
ada.fit(x_train,y_train)

y_pred = ada.predict(x_test)

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n--- Adaboost Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {pre * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("confusion Matrix")
cm = confusion_matrix(y_test,y_pred)

cm_df = pd.DataFrame( cm, index= [f"Actual {c}" for c in ada.classes_],
                      columns=[f"predicted {c}" for c in ada.classes_])

print(cm_df.to_string())


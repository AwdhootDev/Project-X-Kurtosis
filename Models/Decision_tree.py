import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score,recall_score

df_train = pd.read_csv("../Output/train_features.csv")
df_test = pd.read_csv("../Output/test_features.csv")

x_train = df_train.drop(columns=['label', 'file_name'])
y_train = df_train['label']

# print(df.columns.to_list())

x_test = df_test.drop(columns=['label', 'file_name'])
y_test = df_test['label']

model = DecisionTreeClassifier(max_depth=5, random_state=42)
print("training model")
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

acc =  accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average = 'weighted')
rec = recall_score(y_test,y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')

print("\n--- Decision_tree Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, 
    index=[f'Actual {c}' for c in model.classes_], 
    columns=[f'Predicted {c}' for c in model.classes_]
)

print(cm_df.to_string())
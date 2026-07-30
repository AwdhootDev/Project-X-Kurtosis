import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

df_train = pd.read_csv("Output/train_features.csv")
df_test = pd.read_csv("Output/test_features.csv")

x_train = df_train.drop(columns=["label"] + ["file_name"], axis=1)
y_train = df_train["label"]

x_test = df_test.drop(columns=["label"] + ["file_name"], axis=1)
y_test = df_test["label"]

# print(x.shape)
# print(y.shape)

# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# print("Classes : ", encoder.classes_)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# print("X : ", x_train.shape)
# print("X : ", x_test.shape)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1, max_depth = 3, random_state=42)

gb.fit(x_train, y_train)
y_pred = gb.predict(x_test)

print("Accuracy : ", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print()
print(confusion_matrix(y_test, y_pred))
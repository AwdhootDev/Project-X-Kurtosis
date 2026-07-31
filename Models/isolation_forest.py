import pandas as pd 
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

df_train = pd.read_csv("Output/train_features.csv")
df_test = pd.read_csv("Output/test_features.csv")

df_train = df_train[df_train["label"] == 0]

x_train = df_train.drop(columns=["label"] + ["file_name"], axis=1)
# y_train = df_train["label"]

# y_test = df_test["label"]
# y = df["label"]

# print(x.shape)
# print(y.shape)

# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# print("X : ", x_train.shape)
# print("X : ", x_test.shape)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)

# x_test = scaler.transform(x_test)

isof = IsolationForest(n_estimators = 100, contamination = 0.01, random_state = 42)
isof.fit(x_scaled)

x_test = df_test.drop(columns=["label"] + ["file_name"], axis=1)
x_test = scaler.transform(x_test)

y_pred = isof.predict(x_test)

df_test["Prediction"] = y_pred

df_test["binary_label"] = df_test["label"].apply(
    lambda x: 1 if x == 0 else -1
)

print(df_test["Prediction"].value_counts())
print(df_test["binary_label"].value_counts())
print("Accuracy : ", accuracy_score(df_test["binary_label"], y_pred)*100)
print(classification_report(df_test["binary_label"], y_pred))
print("", confusion_matrix(df_test["binary_label"], y_pred))
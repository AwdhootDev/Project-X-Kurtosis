import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df_train = pd.read_csv("Output/train_features.csv")
df_test = pd.read_csv("Output/test_features.csv")
# # print(df.head())
# print(df.columns.to_list())

# feature_drop = ("label", "file_name")

x_train = df_train.drop(columns=["label"] + ["file_name"], axis=1)
y_train = df_train["label"]

x_test = df_test.drop(columns=["label"] + ["file_name"], axis=1)
y_test = df_test["label"]

# # print(x.shape)
# # print(y.shape)

# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# print("Classes : ", encoder.classes_)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# print("X : ", x_train.shape)
# print("X : ", x_test.shape)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 10, n_jobs=-1)

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

acc = accuracy_score(y_test, y_pred) 
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n--- Random Forest Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

# print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
# print()
# print("Classification Report\n", classification_report(y_test, y_pred))
# print()
print("\nConfusion Matrix\n", confusion_matrix(y_test, y_pred))

importance = pd.DataFrame({
    "feature": x_train.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(by = "Importance", ascending = False)

print(importance)

plt.figure(figsize=(8, 5))
plt.bar(importance["feature"], importance["Importance"])
plt.xticks(rotation=45)
plt.ylabel("Importance")
plt.title("Random Forest feature Importance")
plt.tight_layout()
plt.show()

import pandas as pd 
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df_train = pd.read_csv("../Output/train_features.csv")
df_test = pd.read_csv("../Output/test_features.csv")

x_train = df_train.drop(columns=["label"] + ["file_name"])
y_train = df_train["label"]

x_test = df_test.drop(columns=["label"] + ["file_name"])
y_test = df_test["label"]

# print(x.shape)
# print(y.shape)

# encoder = LabelEncoder()
# y = encoder.fit_transform(y)

# print("Classes : ", encoder.classes_)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# print("X : ", x_train.shape)
# print("X : ", x_test.shape)

xgb = XGBClassifier(n_estimators=100, learning_rate = 0.1, max_depth = 7, random_state=42, eval_metric="mlogloss")

xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)

print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
print()
print(classification_report(y_test, y_pred))
print()
print(confusion_matrix(y_test, y_pred))

def fix_right_margin(ax=None, pad_frac=0.18):
    ax = ax or plt.gca()
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * pad_frac)

print("\n--- Starting Global SHAP Analysis (All Classes Combined) ---")

explainer = shap.TreeExplainer(xgb)
shap_values_raw = explainer.shap_values(x_test)

# Aggregation A: mean(|SHAP|) across classes -> bar plot
if isinstance(shap_values_raw, list):
    stacked_shap = np.stack(shap_values_raw, axis=-1)
    global_shap_matrix = np.mean(np.abs(stacked_shap), axis=-1)
elif np.array(shap_values_raw).ndim == 3:
    global_shap_matrix = np.mean(np.abs(shap_values_raw), axis=-1)
else:
    global_shap_matrix = np.abs(shap_values_raw)

global_explanation = shap.Explanation(
    values=global_shap_matrix,
    data=x_test.values,
    feature_names=x_test.columns
)

preds = np.asarray(xgb.predict(x_test)).astype(int)

if isinstance(shap_values_raw, list):
    shap_values_signed = np.array([shap_values_raw[c][i] for i, c in enumerate(preds)])
elif np.array(shap_values_raw).ndim == 3:
    shap_values_signed = np.array([shap_values_raw[i, :, c] for i, c in enumerate(preds)])
else:
    shap_values_signed = shap_values_raw

if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_values_signed = np.array([explainer.expected_value[c] for c in preds])
else:
    base_values_signed = explainer.expected_value

print("Generating Global SHAP Bar Plot...")
plt.figure(figsize=(12, 7))
shap.plots.bar(global_explanation, max_display=15, show=False)
plt.title("Global SHAP Feature Importance (|SHAP| across Classes)", fontsize=14, pad=15)
fix_right_margin(pad_frac=0.18)     
plt.subplots_adjust(left=0.35, right=0.92, top=0.90, bottom=0.10)
plt.show()

print("Generating Global SHAP Beeswarm Plot...")
plt.figure(figsize=(12, 7))
shap.summary_plot(
    shap_values_signed,
    x_test,
    max_display=15,
    show=False
)
plt.title("Global SHAP Beeswarm Plot (Predicted-Class SHAP Values)", fontsize=14, pad=15)
fix_right_margin(pad_frac=0.10)
plt.subplots_adjust(left=0.35, right=0.90, top=0.90, bottom=0.10)
plt.show()

print("Generating SHAP Decision Plot...")
plt.figure(figsize=(12, 7))
shap.decision_plot(
    base_values_signed.mean() if isinstance(base_values_signed, np.ndarray) else base_values_signed,
    shap_values_signed,
    x_test,
    feature_names=list(x_test.columns),
    show=False,
    auto_size_plot=False  
)
plt.title("SHAP Decision Plot (All Samples)", fontsize=14, pad=15)
fix_right_margin(pad_frac=0.10)
plt.subplots_adjust(left=0.35, right=0.90, top=0.88, bottom=0.10)  
plt.show()
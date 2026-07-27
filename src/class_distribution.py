import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("Output/features.csv")

class_counts = df["label"].value_counts()

plt.figure(figsize = (8, 5))
class_counts.plot(kind = "bar")

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("number of Samples")
plt.xticks(rotation = 0)

plt.savefig("Output/plots/class_distribution.png", dpi=100)
plt.show()
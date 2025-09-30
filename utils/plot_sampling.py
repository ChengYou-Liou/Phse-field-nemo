import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("allen_cahn_rar.csv")

#print(df.columns.tolist())
x = df["Points:0"]
t = df["t"]

plt.figure(figsize=(5,5))
plt.scatter(x, t, color='grey', label="PDE residual", s=3)
plt.xlabel("x")
plt.ylabel("t")
plt.title("Scatter Plot: t vs x")
plt.legend()
plt.grid(True)
#plt.axis('off')
plt.savefig("t_vs_x.png", dpi=300,bbox_inches='tight', pad_inches=0, transparent=True)
import numpy as np
import pandas as pd

# Load CSV containing true and predicted phi
df = pd.read_csv("rar/validator500k.csv")
df.rename(columns={"Points:0": "x"}, inplace=True)

# Extract predicted and true phi columns
phi_pred = df["pred_phi"].values   # replace with your prediction column
phi_true = df["true_phi"].values  # replace with your true column

# Compute MSE
mse = np.mean((phi_true - phi_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse:.6e}")

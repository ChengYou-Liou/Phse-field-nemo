import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load CSV
df = pd.read_csv("rar/validator500k.csv")
df.rename(columns={"Points:0": "x"}, inplace=True)

# Extract variables
x = df["x"].values
t = df["t"].values
phi_pred = df["pred_phi"].values  # predicted phase field

# Define regular mesh
grid_size = 200  # number of points in each direction
xi = np.linspace(x.min(), x.max(), grid_size)
ti = np.linspace(t.min(), t.max(), grid_size)
X, T = np.meshgrid(xi, ti, indexing="ij")  # X.shape = T.shape = (grid_size, grid_size)

# Interpolate phi_pred onto the mesh
PHI_grid = griddata(
    points=(x, t),
    values=phi_pred,
    xi=(X, T),
    method="linear"  # options: 'nearest', 'linear', 'cubic'
)

# Plot interpolated 2D field
plt.figure(figsize=(6,5), dpi=100)
plt.imshow(
    PHI_grid.T,  # transpose so t is vertical axis
    origin='lower',
    extent=[x.min(), x.max(), t.min(), t.max()],
    cmap='coolwarm',
    aspect='auto',
    vmin=0, vmax=1
)
plt.xlabel("x")
plt.ylabel("t")
plt.title("Predicted Phase Field (phi)")
plt.colorbar(label="phi")

#Load the sampled points
df_points = pd.read_csv("rar/ac_rar500k.csv")
df_points.rename(columns={"Points:0": "x"}, inplace=True)
x_pts = df_points["x"].values
t_pts = df_points["t"].values

plt.scatter(x_pts, t_pts, color='grey', s=3)  # overlay scatter points

df_rar = pd.read_csv("rar/ac500k.csv")
df_rar.rename(columns={"Points:0": "x"}, inplace=True)
x_rar = df_rar["x"].values
t_rar = df_rar["t"].values

plt.scatter(x_rar, t_rar, color='green', s=3)  # overlay scatter points

plt.tight_layout()
plt.savefig("phi_with_scatter.png", dpi=300)


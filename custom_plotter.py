import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

class CorrosionPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        """
        Plots the predicted, true, and error phase field (phi) values
        on a 2D x vs. t plot.
        """
        x = invar["x"][:, 0]
        t = invar["t"][:, 0]

        
        # Determine the plot extent
        x_min, x_max = x.min(), x.max()
        t_min, t_max = t.min(), t.max()
        
        phi_pred = pred_outvar["phi"][:, 0]

        f, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        f.suptitle("1D Diffusion-Controlled Corrosion", fontsize=14)

        # Plot predicted phase field
        pred = axes[0].scatter(x, t, c=phi_pred, cmap="coolwarm", vmin=0, vmax=1)
        axes[0].set_title("Predicted Phase Field")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(t_min, t_max)
        f.colorbar(pred, ax=axes[0], label="phi")
        
        # Plot true and error phase fields if available
        if true_outvar is not None and "phi" in true_outvar:
            phi_true = true_outvar["phi"][:, 0]
            
            # Plot true phase field
            true = axes[1].scatter(x, t, c=phi_true, cmap="coolwarm", vmin=0, vmax=1)
            axes[1].set_title("True Phase Field")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("t")
            axes[1].set_xlim(x_min, x_max)
            axes[1].set_ylim(t_min, t_max)
            f.colorbar(true, ax=axes[1], label="phi")
            
            # Plot absolute error
            diff = np.abs(phi_pred - phi_true)
            err = axes[2].scatter(x, t, c=diff, cmap="coolwarm", vmin=0, vmax=1)
            axes[2].set_title("Absolute Error")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("t")
            axes[2].set_xlim(x_min, x_max)
            axes[2].set_ylim(t_min, t_max)
            f.colorbar(err, ax=axes[2], label="abs error")
        else:
            # If no true data, hide the extra plots
            axes[1].axis('off')
            axes[2].axis('off')

        plt.tight_layout()
        plt.show()
        
        return [(f, "corrosion_results")]
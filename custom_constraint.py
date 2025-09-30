import torch
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.domain.constraint import (
    Constraint,
    PointwiseInteriorConstraint,
)

class RarInteriorConstraint(PointwiseInteriorConstraint):
    def save_batch(self, filename):
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.device)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=self.device)

        # If using DDP, strip out collective stuff to prevent deadlocks
        # This only works either when one process alone calls in to save_batch
        # or when multiple processes independently save data
        if hasattr(self.model, "module"):
            modl = self.model.module
        else:
            modl = self.model

        # compute pred outvar
        pred_outvar = modl(invar)

        if "Allen-Cahn" in pred_outvar:
            residual = torch.abs(pred_outvar["Allen-Cahn"]).view(-1)
            #pick indicies of top 4000
            topk = min(4000, residual.numel())
            _, idx = torch.topk(residual, k = topk, largest=True)

            invar = {k: v[idx] for k, v in invar.items()}
            true_outvar = {k: v[idx] for k, v in true_outvar.items()}
            pred_outvar = {k: v[idx] for k, v in pred_outvar.items()}
            lambda_weighting = {k: v[idx] for k, v in lambda_weighting.items()}

        # rename values and save batch to vtk file TODO clean this up after graph unroll stuff
        named_lambda_weighting = {
            "lambda_" + key: value for key, value in lambda_weighting.items()
        }
        named_true_outvar = {"true_" + key: value for key, value in true_outvar.items()}
        named_pred_outvar = {"pred_" + key: value for key, value in pred_outvar.items()}
        save_var = {
            **{key: value for key, value in invar.items()},
            **named_true_outvar,
            **named_pred_outvar,
            **named_lambda_weighting,
        }
        save_var = {
            key: value.cpu().detach().numpy() for key, value in save_var.items()
        }
        var_to_polyvtk(save_var, filename)
import torch
from torch import Tensor
from typing import Dict
from physicsnemo.sym.loss import Loss

class TopKLoss(Loss):
    def __init__(self, ord: int = 2):
        super().__init__()
        self.ord: int = ord

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
        ord: float,
        topk_pde: int = 4000,
    ) -> Dict[str, Tensor]:
        losses = {}
        
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * torch.abs(
                pred_outvar[key] - true_outvar[key]
            ).pow(ord)
            if "area" in invar.keys():
                l *= invar["area"]

            if key.lower() == "allen-cahn" or key.lower() == "cahn-hilliard":
                l_flat = l.flatten()
                if l_flat.numel() > topk_pde:
                    topk_values, _ = torch.topk(l_flat, topk_pde)
                    l = topk_values
                else:
                    l = l_flat
                    
            losses[key] = l.sum()
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return TopKLoss._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step, self.ord
        )
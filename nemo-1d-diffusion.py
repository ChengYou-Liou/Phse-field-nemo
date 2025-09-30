import time
import os
import warnings
import sympy
import numpy as np
import torch
from sympy import Symbol, Function
from typing import Dict, List

import physicsnemo.sym
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig, to_absolute_path
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D, Point1D
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.dataset import (
    DictPointwiseDataset,
)
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint,
    PointwiseBoundaryConstraint,
)
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.loss import PointwiseLossNorm

from custom_plotter import CorrosionPlotter
from custom_constraint import RarInteriorConstraint
from TopK import TopKLoss

def importance_measure(invar):
    x = invar['x']
    eps = 1e-4

    importance = 1.0 / (torch.abs(x - 0.4) + eps)
    return importance

class CahnHilliard(PDE):
    name="CahnHilliard"

    def __init__(self, AA=1.0e7, DD=8.5e-10, CSE=1.0, CLE=5100 / 1.43e5, GEO_COEF=1e4, TIME_COEF=1e-2, dim=3, time=False):
        self.dim = dim
        self.time = time
        self.AA = AA
        self.DD = DD
        self.CSE = CSE
        self.CLE = CLE

        x = Symbol('x')
        y = Symbol('y')
        z = Symbol('z')
        t = Symbol('t')

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        elif self.dim == 1:
            input_variables.pop("z")
            input_variables.pop("y")

        if not self.time:
            input_variables.pop("t")

        c = Function("c")(*input_variables)
        phi = Function("phi")(*input_variables)

        nabla2c = (c.diff(x, 2) + c.diff(y, 2) + c.diff(z, 2)) * (GEO_COEF ** 2)
        nabla2phi = (phi.diff(x, 2) + phi.diff(y, 2) + phi.diff(z, 2)) * (GEO_COEF ** 2)
        nabla2_df_dc = 2 * AA * (nabla2c + 6 * (CSE - CLE) * 
                                 (phi * (phi - 1) * nabla2phi 
                                    + (2 * phi - 1) * (phi.diff(x)**2 + phi.diff(y)**2 + phi.diff(z)**2) * (GEO_COEF**2)
                                )
        )

        self.equations = {
            "Cahn-Hilliard": c.diff(t) * TIME_COEF - DD / 2 / AA * nabla2_df_dc,
        }

class AllenCahn(PDE):
    name="AllenCahn"

    def __init__(self, LP=2.0, AA=5.35e7, CSE=1.0, CLE=5100 / 1.43e5, ALPHA_PHI=1.03e-1, OMEGA_PHI=1.76e7, GEO_COEF=1e4, TIME_COEF=1e-2, dim=3, time=False):
        self.dim = dim
        self.time = time

        x = Symbol('x')
        y = Symbol('y')
        z = Symbol('z')
        t = Symbol('t')

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        elif self.dim == 1:
            input_variables.pop("z")
            input_variables.pop("y")

        if not self.time:
            input_variables.pop("t")

        phi = Function("phi")(*input_variables)
        c = Function("c")(*input_variables)

        df_dphi = 12 * AA * (CSE - CLE) * phi * (phi - 1) * \
            (c - (CSE - CLE) * (-2 * phi**3 + 3 * phi**2) - CLE) \
            + 2 * OMEGA_PHI * phi * (phi - 1) * (2 * phi - 1)

        self.equations = {
            "Allen-Cahn": phi.diff(t) * TIME_COEF + LP * (df_dphi - ALPHA_PHI * (GEO_COEF**2) * (phi.diff(x, 2) + phi.diff(y, 2) + phi.diff(z, 2))),
        }

def phi_initial_condition(x, alpha_phi, omega_phi, GEO_COEF):
    if isinstance(x, sympy.Expr):
        mathlib = sympy
    else:
        mathlib = np
    phi_val = 0.5 - 0.5 * mathlib.tanh(mathlib.sqrt(omega_phi) / mathlib.sqrt(2 * alpha_phi) * (x - 0.40) / GEO_COEF)
    return phi_val

def c_initial_condition(x, alpha_phi, omega_phi, CSE, GEO_COEF):
    if isinstance(x, sympy.Expr):
        mathlib = sympy
    else:
        mathlib = np
    phi = 0.5 - 0.5 * mathlib.tanh(mathlib.sqrt(omega_phi) / mathlib.sqrt(2 * alpha_phi) * (x - 0.40) / GEO_COEF)
    c_val = (-2 * phi**3 + 3 * phi**2) * CSE
    return c_val

#parameters
ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
AA = 5.35e7
DD = 8.5e-10
LP = 2.0
CSE = 1.0
CLE = 5100 / 1.43e5

GEO_COEF = 1e4
TIME_COEF = 1e-2

@physicsnemo.sym.main(config_path="conf", config_name="nemo-1d-diffusion")
def run (cfg: PhysicsNeMoConfig) ->None:
    x, t = Symbol('x'), Symbol('t')
    time_range = {t: (0, 1.0)}

    #Set up PDE and network
    ch = CahnHilliard(AA=AA, DD=DD, CSE=CSE, CLE=CLE, GEO_COEF=GEO_COEF, TIME_COEF=TIME_COEF, dim=1, time=True)
    ac = AllenCahn(LP=LP, AA=AA, CSE=CSE, CLE=CLE, ALPHA_PHI=ALPHA_PHI, OMEGA_PHI=OMEGA_PHI, GEO_COEF=GEO_COEF, TIME_COEF=TIME_COEF, dim=1, time=True)

    FC = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("phi"), Key("c")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        ch.make_nodes()
        + ac.make_nodes()
        + [FC.make_node(name="FC")]
    )

    invar = [Key("x"), Key("t")]
    pde_output_keys = [Key("Cahn-Hilliard"), Key("Allen-Cahn")]
    graph = Graph(nodes=nodes, invar=invar, req_names=pde_output_keys)
    
    #make geometry
    geo = Line1D(-0.5, 0.5)
    local = Line1D(0.3, 0.5)
    left = Point1D(-0.5)
    right = Point1D(0.5)
    boundary = left + right

    #make domain
    domain = Domain()

    #Constraint
    interior_ch = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"Cahn-Hilliard": 0},
        batch_size=cfg.batch_size.cahn_hilliard,
        parameterization=time_range,
        lambda_weighting = {
            "Cahn-Hilliard" : 1e18,
        },
    )
    domain.add_constraint(interior_ch, "interior_ch")

    interior_ac_global = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"Allen-Cahn": 0},
        batch_size=cfg.batch_size.allen_cahn_global,
        parameterization=time_range,
        lambda_weighting = {
            "Allen-Cahn":1.0,
        },
    )
    domain.add_constraint(interior_ac_global, "interior_ac_global")

    interior_rar = RarInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        dataset=None,
        outvar={"Allen-Cahn":0, "Cahn-Hilliard":0,},
        parameterization=time_range,
        batch_size=4000,
        fixed_dataset=True,
        lambda_weighting={
            "Allen-Cahn":1.0,
            "Cahn-Hilliard": 5.0e15,
        },
        loss=TopKLoss(),
    )
    domain.add_constraint(interior_rar, "interior_rar")

    #initial condition
    IC_phi = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"phi": phi_initial_condition(x, ALPHA_PHI, OMEGA_PHI, GEO_COEF)},
        batch_size=cfg.batch_size.initial_condition_phi,
        parameterization={t: 0.0},
        importance_measure = importance_measure,
        num_workers = 8,
        lambda_weighting = {
            "phi":5.0e15,
        },
    )
    domain.add_constraint(IC_phi, "IC_phi")

    IC_local = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=local,
        outvar={"phi": phi_initial_condition(x, ALPHA_PHI, OMEGA_PHI, GEO_COEF)},
        batch_size=cfg.batch_size.initial_condition_phi,
        fixed_dataset=True,
        num_workers=8,
        parameterization={t: 0.0},
        bounds={"x": (0.3, 0.5), "t": (0.0, 0.0)},
        lambda_weighting = {
            "phi":5.0e15,
        },
    )
    domain.add_constraint(IC_local, "IC_local")

    IC_c = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"c": c_initial_condition(x, ALPHA_PHI, OMEGA_PHI, CSE, GEO_COEF)},
        batch_size=cfg.batch_size.initial_condition_c,
        parameterization={t: 0.0},
        lambda_weighting={"c": 8.0e14}
    )
    domain.add_constraint(IC_c, "IC_c")

    #boundary condition
    BC_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=left,
        outvar={"phi": 1.0, "c": 1.0},
        batch_size=cfg.batch_size.boundary_condition,
        parameterization=time_range,
        lambda_weighting = {
            "phi":1.0e16,
            "c": 8.0e14,
        },
    )
    domain.add_constraint(BC_left, "BC_left")

    BC_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=right,
        outvar={"phi": 0.0, "c": 0.0},
        batch_size=cfg.batch_size.boundary_condition,
        parameterization=time_range,
        lambda_weighting = {
            "phi":1.0e16,
            "c": 8.0e14,
        },
    )
    domain.add_constraint(BC_right, "BC_right")

    #add validation data
    file_path = "data/results-fenics-diffusion.csv"
    if os.path.exists(to_absolute_path(file_path)):
        pass
    else:
        warnings.warn(f"Validation dataset not found at {file_path}. Continuing without validation data.")

    fenics_var = csv_to_dict(
        to_absolute_path(file_path), {"x": "x", "t": "t", "phi": "phi", "c": "c"}
    )

    fenics_var["t"] *= TIME_COEF
    fenics_var["x"] *= GEO_COEF

    fenics_invar = {
        key: value for key, value in fenics_var.items() if key in ["x", "t"]
    }
    fenics_outvar = {
        key: value for key, value in fenics_var.items() if key in ["phi", "c"]
    }

    plotter = CorrosionPlotter()

    fenics_validator = PointwiseValidator(
        nodes=nodes,
        invar=fenics_invar,
        true_outvar=fenics_outvar,
        batch_size=cfg.batch_size.validation,
        plotter=plotter
    )
    domain.add_validator(fenics_validator)


    #make solver
    slv = Solver(cfg, domain)

    #start solver
    slv.solve()

if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
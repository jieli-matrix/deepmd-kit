# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    List,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
    GeneralFitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("density")
class DensityFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: List[int] = [128, 128, 128],
        grid_embedding_neurons: List[int] = [],  # Add this line
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: Optional[Union[int, List[int]]] = None,
        type_map: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            "density",
            ntypes,
            dim_descrpt,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            type_map=type_map,
            **kwargs,
        )
        self.grid_embedding_neurons = grid_embedding_neurons  

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data.pop("var_name")
        data.pop("dim_out")
        grid_embedding_neurons = data.pop("grid_embedding_neurons", [])  
        instance = super().deserialize(data)
        instance.grid_embedding_neurons = grid_embedding_neurons  
        return instance

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "density",
            "grid_embedding_neurons": self.grid_embedding_neurons, 
        }

    # make jit happy with torch 2.0.0
    exclude_types: List[int]

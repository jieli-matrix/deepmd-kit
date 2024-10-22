# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import torch

from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    RepformerLayer,
    _make_nei_g1,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.task.density import (
    DensityFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)

from .dp_atomic_model import (
    DPAtomicModel,
)

log = logging.getLogger(__name__)


class DPDensityAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, DensityFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)
        self.rcut = self.descriptor.get_rcut()
        self.rcut_smth = self.descriptor.get_rcut_smth()
        self.env_protection = self.descriptor.get_env_protection()
        if self.env_protection == 0.0:
            self.env_protection = 1e-6
        self.sel = self.descriptor.get_sel()
        self.nnei = self.descriptor.get_nsel()
        self.axis_neuron = self.descriptor.axis_neuron

        # Get the grid_embedding_neurons from the fitting instance
        self.grid_embedding_neurons = fitting.grid_embedding_neurons if hasattr(fitting, 'grid_embedding_neurons') else []
        
        dims = [1 + self.descriptor.repinit_args.tebd_dim] + self.grid_embedding_neurons + [self.descriptor.get_dim_out()]
        self.grid_embedding_layers = [MLPLayer(
            dims[i],
            dims[i+1],
            precision=env.DEFAULT_PRECISION,
            activation_function="tanh",
        ) for i in range(len(self.grid_embedding_neurons) + 1)]

        wanted_shape = (1, self.nnei, 4)
        mean = torch.zeros(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        stddev = torch.ones(
            wanted_shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
        grid: Optional[torch.Tensor] = None,
        grid_type: Optional[torch.Tensor] = None,
        grid_nlist: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
        mapping
            mapps the extended indices to local indices
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
        )
        assert descriptor is not None
        assert grid is not None
        assert grid_type is not None
        assert grid_nlist is not None
        bsz, ngrid, nnei = grid_nlist.shape
        grid_nlist_mask = grid_nlist >= 0
        merged_coord = torch.cat([grid, extended_coord], dim=1)
        shifted_nlist = torch.where(grid_nlist_mask, grid_nlist + ngrid, -1)
        dmatrix, diff, sw = prod_env_mat(
            merged_coord,
            shifted_nlist,
            grid_type,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )
        # 1. nb x ngrid x nnei x 3
        h2 = diff / self.rcut
        # 2. nb x ngrid x nnei x 4
        h2 = dmatrix
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        assert mapping is not None  # need fix for comm_dict
        mapping_ng1 = (
            mapping.view(nframes, nall)
            .unsqueeze(-1)
            .expand(-1, -1, self.descriptor.get_dim_out())
        )
        # nb x nall x ng1
        g1_ext = torch.gather(descriptor, 1, mapping_ng1)

        ng1 = self.descriptor.get_dim_out()
        # nb x nall x ntebd
        tebd = self.descriptor.type_embedding(extended_atype)
        ntebd = tebd.shape[-1]
        grid_nlist_0 = torch.where(grid_nlist_mask, grid_nlist, 0)
        grid_nlist_expanded = grid_nlist_0.view(nframes, ngrid*nnei).unsqueeze(-1).expand(-1, -1, ntebd)
        # nb x ngrid x nnei x ntebd
        grid_tebd = torch.gather(tebd, 1, grid_nlist_expanded).view(nframes, ngrid, nnei, ntebd)
        grid_tebd = torch.where(grid_nlist_mask.unsqueeze(-1), grid_tebd, 0)
        # nb x ngrid x nnei x (1+ntebd)
        h2_and_type = torch.concat([h2[:, :, :, :1], grid_tebd], -1)
        # nb x ngrid x nnei x ng1
        gg = h2_and_type
        for layer in self.grid_embedding_layers:
            gg = layer(gg)
        # electron-to-atom equivariant feature: nb x ngrid x nnei x 4 x ng1
        e2aef = h2.unsqueeze(-1) * gg.unsqueeze(-2)

        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.descriptor.repinit.mean,
            self.descriptor.repinit.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )
        # nb x nloc x nnei x 4
        h2 = dmatrix
        nlist_mask = nlist >= 0
        nlist_0 = torch.where(nlist_mask, nlist, 0)
        # nb x nloc x nnei x ng1
        gg1 = _make_nei_g1(g1_ext, nlist_0)
        # nb x nloc x 4 x ng1
        h2g1 = RepformerLayer._cal_hg(
            gg1, h2, nlist_mask, sw.squeeze(-1), smooth=True, epsilon=1e-4
        )

        grid_nlist_0 = torch.where(grid_nlist_mask, grid_nlist, 0)
        # nb x (ngrid*nnei)
        grid_nlist_loc = torch.gather(mapping, 1, grid_nlist_0.view(nframes, ngrid*nnei))
        # nb x (ngrid*nnei) x 4 x ng1
        grid_nlist_expanded = grid_nlist_loc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, ng1)
        # atomic equivariant feature: nb x ngrid x nnei x 4 x ng1
        aef = torch.gather(h2g1, 1, grid_nlist_expanded).view(nframes, ngrid, nnei, 4, ng1)
        aef = torch.where(grid_nlist_mask.unsqueeze(-1).unsqueeze(-1), aef, 0)
        # electron-to-atom invariant feature: nb x ngrid x nnei x ng1
        e2aif = torch.sum(e2aef*aef, -2)

        # atomic invariant feature: nb x ngrid x nnei x ng1
        grid_nlist_expanded = grid_nlist_loc.unsqueeze(-1).expand(-1, -1, ng1)
        aif = torch.gather(descriptor, 1, grid_nlist_expanded).view(nframes, ngrid, nnei, ng1)
        aif = torch.where(grid_nlist_mask.unsqueeze(-1), aif, 0)
        # electron-and-atom invariant feature: nb x ngrid x nnei x (ng1*2)
        new_descriptor = torch.concat([aif, e2aif], 3)

        fit_ret = self.fitting_net(
            new_descriptor.view(nframes, ngrid*nnei, ng1*2),
            torch.zeros([nframes, ngrid*nnei], device=grid_type.device, dtype=grid_type.dtype),
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        # nb x ngrid x nnei x 1
        nei_density = torch.exp(fit_ret["density"].view(nframes, ngrid, nnei, 1))
        nei_density = torch.where(grid_nlist_mask.unsqueeze(-1), nei_density, 0)
        # nb x ngrid x 1
        grid_density = torch.sum(nei_density, -2)
        ret = {
            "density": grid_density,
        }
        return ret

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
        grid: Optional[torch.Tensor] = None,
        grid_type: Optional[torch.Tensor] = None,
        grid_nlist: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Common interface for atomic inference.

        This method accept extended coordinates, extended atom typs, neighbor list,
        and predict the atomic contribution of the fit property.

        Parameters
        ----------
        extended_coord
            extended coodinates, shape: nf x (nall x 3)
        extended_atype
            extended atom typs, shape: nf x nall
            for a type < 0 indicating the atomic is virtual.
        nlist
            neighbor list, shape: nf x nloc x nsel
        mapping
            extended to local index mapping, shape: nf x nall
        fparam
            frame parameters, shape: nf x dim_fparam
        aparam
            atomic parameter, shape: nf x nloc x dim_aparam
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        ret_dict
            dict of output atomic properties.
            should implement the definition of `fitting_output_def`.
            ret_dict["mask"] of shape nf x nloc will be provided.
            ret_dict["mask"][ff,ii] == 1 indicating the ii-th atom of the ff-th frame is real.
            ret_dict["mask"][ff,ii] == 0 indicating the ii-th atom of the ff-th frame is virtual.

        """
        assert grid is not None
        assert grid_type is not None
        assert grid_nlist is not None
        _, nloc, _ = nlist.shape
        _, ngrid, _ = grid_nlist.shape
        atype = extended_atype[:, :nloc]

        if self.pair_excl is not None:
            pair_mask = self.pair_excl(nlist, extended_atype)
            # exclude neighbors in the nlist
            nlist = torch.where(pair_mask == 1, nlist, -1)

        ext_atom_mask = self.make_atom_mask(extended_atype)
        ret_dict = self.forward_atomic(
            extended_coord,
            torch.where(ext_atom_mask, extended_atype, 0),
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
            grid=grid,
            grid_type=grid_type,
            grid_nlist=grid_nlist,
        )
        ret_dict = self.apply_out_stat(ret_dict, grid_type)

        ext_grid_mask = self.make_atom_mask(grid_type)

        # nf x ngrid
        grid_mask = ext_atom_mask[:, :ngrid].to(torch.int32)
        if self.atom_excl is not None:
            grid_mask *= self.atom_excl(grid_type)

        for kk in ret_dict.keys():
            out_shape = ret_dict[kk].shape
            out_shape2 = 1
            for ss in out_shape[2:]:
                out_shape2 *= ss
            ret_dict[kk] = (
                ret_dict[kk].reshape([out_shape[0], out_shape[1], out_shape2])
                * grid_mask[:, :, None]
            ).view(out_shape)
        ret_dict["mask"] = grid_mask

        return ret_dict

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_common_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
        )

    def compute_or_load_out_stat(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        log.warning("Not implemented yet for density out stat!")
        pass

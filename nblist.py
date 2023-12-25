"""
temporaily, we use freud to calculate neighbor
"""
from typing import Optional, Literal
import torch
import numpy as np
from functorch import vmap
from dmff_torch.utils import jit_condition
from dmff_torch.pairwise import distribute_v3

def build_covalent_map(data, max_neighbor):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_atoms = len(data['positions'])
    covalent_map = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in data['bonds']:
        covalent_map[bond[0],bond[1]] = 1
        covalent_map[bond[1],bond[0]] = 1
    for n_curr in range(1, max_neighbor):
        for i in range(n_atoms):
            # current neighbors
            j_list = np.where(
                np.logical_and(covalent_map[i] <= n_curr,
                               covalent_map[i] > 0))[0]
            for j in j_list:
                k_list = np.where(covalent_map[j] == 1)[0]
                for k in k_list:
                    if k != i and k not in j_list:
                        covalent_map[i, k] = n_curr + 1
                        covalent_map[k, i] = n_curr + 1
    return torch.tensor(covalent_map, dtype=torch.float32, device=device)

def neighborlist(positions, pairs, box, rcut):
    ####################################################
    # only solute-solute interaction or solute-solvent 
    # calculations are considered 
    #################################################### 
    @partial(vmap, in_dims=(0, None, None), out_dims=0)
    @jit_condition()
    def v_pbc_shift(drvecs, box, box_inv):
        unshifted_dsvecs = torch.matmul(drvecs, box_inv)
        dsvecs = unshifted_dsvecs - torch.floor(unshifted_dsvecs + 0.5)
        return torch.matmul(dsvecs, box)
    device = positions.device
    r1 = distribute_v3(positions.T, pairs[:,0]).T
    r2 = distribute_v3(positions.T, pairs[:,1]).T
    dr = r1 - r2
    box_inv = torch.linalg.inv(box)
    dr = v_pbc_shift(dr, box, box_inv)
    mask = dr.pow(2).sum(-1) < rcut**2
    # return the pairs that distance within rcut 
    return pairs[mask]

class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True):
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map

    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = np.concatenate([pairs, nbond[:, None]], axis=1)
        return torch.tensor(pairs)

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.capacity_multiplier is None:
            self.capacity_multiplier = int(nlist.shape[0] * 1.3)

        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions


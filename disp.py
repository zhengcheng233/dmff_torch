#!/usr/bin/env python
"""
my second torch script, use to calculate disp-disp interaction 
"""
from functools import partial
import torch
from dmff_torch.pairwise import distribute_v3, distribute_scalar, distribute_dispcoeff
from dmff_torch.pme import setup_ewald_parameters 
from dmff_torch.recip import pme_recip
from dmff_torch.utils import jit_condition
from functorch import vmap
import numpy as np
from dmff_torch.nblist import build_covalent_map

class ADMPDispPmeForce:
    '''
    This is a convenient wrapper for dispersion PME calculations
    It wrapps all the environment parameters of multipolar PME calculation
    The so called "environment paramters" means parameters that do not need to be differentiable
    '''

    def __init__(self, box, rc, ethresh, pmax, lpme=True):

        self.rc = rc
        self.ethresh = ethresh
        self.pmax = pmax
        # Need a different function for dispersion ??? Need tests
        self.lpme = lpme
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if lpme:
            kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
            self.kappa = kappa
            self.K1 = K1
            self.K2 = K2
            self.K3 = K3
            ###############################################################################
            # modify here, for the torch.jit purpose
            pme_order = torch.tensor(6, dtype=torch.int32, device=self.device)
            bspline_range = torch.arange(-pme_order//2, pme_order//2)
            n_mesh = pme_order**3
            shift_y,shift_x,shift_z = torch.meshgrid(bspline_range, bspline_range, bspline_range,indexing='ij')
            shifts = torch.stack((shift_x,shift_y,shift_z)).transpose(0,3).reshape((1,n_mesh,3))
            self.n_mesh = torch.tensor(n_mesh, dtype=torch.int32, device=self.device)
            self.shifts = shifts.to(torch.float32)
            ##############################################################################
        else:
            self.kappa = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.K1 = torch.tensor(0, dtype=torch.int32, device=self.device)
            self.K2 = torch.tensor(0, dtype=torch.int32, device=self.device)
            self.K3 = torch.tensor(0, dtype=torch.int32, device=self.device)
            self.n_mesh = None
            self.shifts = None

        # setup calculators
        self.refresh_calculators()
        return

    #def generate_get_energy(self):
    # energy_disp_pme(positions, box, pairs,
    #                              c_list, mScales,
    #                              self.kappa, self.K1, self.K2, self.K3, self.pmax,
    #                              self.n_mesh, self.shifts, lpme=self.lpme)

    def update_env(self, attr, val):
        '''
        Update the environment of the calculator
        '''
        setattr(self, attr, val)
        self.refresh_calculators()

    def refresh_calculators(self):
        '''
        refresh the energy and force calculator according to the current environment
        '''
        self.get_energy = energy_disp_pme
        return


def energy_disp_pme(positions, box, pairs,
        c_list, mScales, kappa, K1, K2, K3, pmax,
        n_mesh, shifts, aex, aes, apol, adisp, adhf, 
        b, ldmp, lpme=True):
    '''
    Top level wrapper for dispersion pme

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 3: interacting pair indices and topology distance
        c_list:
            Na * (pmax-4)/2: atomic dispersion coefficients
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        disp_pme_recip_fn:
            function: the reciprocal calculator, see recip.py
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculations
        pmax:
            int array: maximal exponents (p) to compute, e.g., (6, 8, 10)
        lpme:
            bool: whether do pme or not, useful when doing cluster calculations

    Output:
        energy: total dispersion pme energy
    '''
    if lpme is False:
        kappa = torch.tensor(0, dtype=torch.int32, device=positions.device)
      
    ene_real = disp_pme_real(positions, box, pairs, c_list, mScales, kappa, pmax, ldmp, aex, aes, apol, adisp, adhf, b)

    if lpme:
        ene_recip = pme_recip(torch.tensor(6, dtype=torch.int32, device=positions.device), kappa, True, K1, K2, K3, positions, box, c_list[:, 0, None], n_mesh, shifts, torch.tensor(0, dtype=torch.int32, device=positions.device))
        if pmax >= torch.tensor(8, dtype=torch.int32, device=positions.device):
            ene_recip += pme_recip(torch.tensor(8, dtype=torch.int32, device=positions.device), kappa, True, K1, K2, K3, positions, box, c_list[:, 1, None], n_mesh, shifts, torch.tensor(0, dtype=torch.int32, device=positions.device))
        if pmax >= torch.tensor(10, dtype=torch.int32, device=positions.device):
            ene_recip += pme_recip(torch.tensor(10, dtype=torch.int32, device=positions.device), kappa, True, K1, K2, K3, positions, box, c_list[:, 2, None], n_mesh, shifts, torch.tensor(0, dtype=torch.int32, device=positions.device))
        ene_self = disp_pme_self(c_list, kappa, pmax)
        return ene_real + ene_recip + ene_self

    else:
        return ene_real

def disp_pme_real(positions, box, pairs,
        c_list,
        mScales,
        kappa, pmax, ldmp, aex, 
        aes, apol, adisp, adhf, b):
    '''
    This function calculates the dispersion real space energy
    It expands the atomic parameters to pairwise parameters

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 3: interacting pair indices and topology distance
        c_list:
            Na * (pmax-4)/2: atomic dispersion coefficients
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        pmax:
            int array: maximal exponents (p) to compute, e.g., (6, 8, 10)

    Output:
        ene: dispersion pme realspace energy
    '''    
    @vmap
    @jit_condition()
    def regularize_pairs(p):
        # using vmap; we view 2-d array with only its element (1-d array, exampe p = p[m]), but dp is same as  p[:,0] - p[:,1]
        dp = p[1] - p[0]
        dp = torch.where(dp > torch.tensor(0, dtype=torch.int32, device=dp.device), torch.tensor(0, dtype=torch.int32, device=dp.device), torch.tensor(1, dtype=torch.int32, device=dp.device))
        # vmap don't support .item on a Tensor, for nopbc system, no buffer atoms 
        #dp_vec = torch.tensor([dp, 2 * dp])
        p[0] = p[0] - dp
        p[1] = p[1] - dp * 2
        return p

    @vmap
    @jit_condition()
    def pair_buffer_scales(p):
        dp = p[0] - p[1]
        return torch.where(dp < torch.tensor(0, dtype=torch.int32, device=dp.device), torch.tensor(1, dtype=torch.int32, device=dp.device), torch.tensor(0, dtype=torch.int32, device=dp.device))

    @partial(vmap, in_dims=(0, None, None), out_dims=0)
    @jit_condition()
    def v_pbc_shift(drvecs, box, box_inv):
        unshifted_dsvecs = torch.matmul(drvecs, box_inv)
        dsvecs = unshifted_dsvecs - torch.floor(unshifted_dsvecs + 0.5)
        return torch.matmul(dsvecs, box)

    pairs[:,:2] = regularize_pairs(pairs[:,:2])

    box_inv = torch.linalg.inv(box)
    
    ri = distribute_v3(positions.T, pairs[:, 0]).T
    rj = distribute_v3(positions.T, pairs[:, 1]).T
    
    nbonds = pairs[:, 2] 
    indices = (nbonds + (mScales.shape[0] - 1)) % mScales.shape[0]
    mscales = distribute_scalar(mScales, indices)

    buffer_scales = pair_buffer_scales(pairs[:, :2])
    mscales = mscales * buffer_scales

    ci = distribute_dispcoeff(c_list.T, pairs[:, 0]).T
    cj = distribute_dispcoeff(c_list.T, pairs[:, 1]).T
 
    dr = ri - rj 
    dr = v_pbc_shift(dr, box, box_inv)
    norm_dr = torch.linalg.norm(dr, dim=-1)

    if ldmp == True:
        aexi = distribute_scalar(aex, pairs[:,0])
        aexj = distribute_scalar(aex, pairs[:,1])
        aesi = distribute_scalar(aes, pairs[:,0])
        aesj = distribute_scalar(aes, pairs[:,1])
        apoli = distribute_scalar(apol, pairs[:,0])
        apolj = distribute_scalar(apol, pairs[:,1])
        adispi = distribute_scalar(adisp, pairs[:,0])
        adispj = distribute_scalar(adisp, pairs[:,1])
        adhfi = distribute_scalar(adhf, pairs[:,0])
        adhfj = distribute_scalar(adhf, pairs[:,1])
        bi = distribute_scalar(b, pairs[:,0])
        bj = distribute_scalar(b, pairs[:,1])
        #################################################################################
        # final disp interaction is - (E_lr - E_sr) here, we give E_lr_real - E_sr
        #################################################################################
        ene_real = - torch.sum(
            disp_dmp_kernel(norm_dr, ci, cj, aexi, aexj, aesi, aesj, apoli, apolj, adispi, adispj, adhfi, adhfj, bi, bj, mscales) * buffer_scales) + torch.sum(disp_pme_real_kernel(norm_dr, ci, cj, box, box_inv, mscales, kappa, pmax) * buffer_scales)

        #ene_real = torch.sum((disp_dmp_kernel(norm_dr, ci, cj, aexi, aexj, aesi, aesj, apoli, apolj, adispi, adispj, adhfi, adhfj, bi, bj, mscales) + 
        #    disp_pme_real_kernel(norm_dr, ci, cj, box, box_inv, mscales, kappa, pmax))
        #    * buffer_scales
        #    )
    else:
        ene_real = torch.sum(
            disp_pme_real_kernel(norm_dr, ci, cj, box, box_inv, mscales, kappa, pmax)
            * buffer_scales
            )
    return torch.sum(ene_real)
    
@partial(vmap, in_dims=(0, None, None), out_dims=0)
def pbc_shift(drvecs, box, box_inv):
    unshifted_dsvecs = torch.matmul(drvecs, box_inv)
    dsvecs = unshifted_dsvecs - torch.floor(unshifted_dsvecs + 0.5)
    return torch.matmul(dsvecs, box)

@jit_condition()
def g_p(x2, pmax):
    '''
    Compute the g(x, p) function

    Inputs:
        x:
            float: the input variable
        pmax:
            int: the maximal powers of dispersion, here we assume evenly spacing even powers starting from 6
            e.g., (6,), (6, 8) or (6, 8, 10)

    Outputs:
        g:
            (p-4)//2: g(x, p)
    '''
    x4 = x2 * x2
    x8 = x4 * x4
    exp_x2 = torch.exp(-x2)
    g6 = (1 + x2 + 0.5 * x4) * exp_x2
    g8 = torch.where(pmax >= torch.tensor(8, dtype=torch.int32, device=x2.device), g6 + (x4 * x2 / 6) * exp_x2, torch.zeros_like(g6))
    g10 = torch.where(pmax >= torch.tensor(10, dtype=torch.int32, device=x2.device), g8 + (x8 / 24) * exp_x2, torch.zeros_like(g6))
    g = [g6, g8, g10]
    return g 

@vmap
#@jit_condition()
def disp_dmp_kernel(dr, ci, cj, aexi, aexj, aesi, aesj, apoli, apolj, adispi, adispj, adhfi, adhfj, bi, bj, m):
    a_ex = (aexi * aexj)
    a_es = (aesi * aesj)
    a_pol = (apoli * apolj)
    a_disp = (adispi * adispj)
    a_dhf = (adhfi * adhfj)
    b = torch.sqrt(bi * bj)
    c6 = ci[0] * cj[0]
    c8 = ci[1] * cj[1]
    c10 = ci[2] * cj[2]
    br = b * dr
    br2 = br * br
    exp_br = torch.exp(-br)
    P = 1/3 * br2 + br + 1
    x = br - (2 * br2 + 3 * br) / (br2 + 3 * br + 3)
    s6 = 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + x**6/720
    s8 = s6 + x**7/5040 + x**8/40320
    s10 = s8 + x**9/362880 + x**10/3628800
    exp_x = torch.exp(-x)
    f6 = exp_x * s6
    f8 = exp_x * s8
    f10 = exp_x * s10
    f = (a_ex + a_es + a_pol + a_disp + a_dhf) * P * exp_br /2625.5 + (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) 
    expdmp = torch.where(dr < torch.tensor(2.5, dtype=torch.float32, device=dr.device), torch.tensor(0., dtype=torch.float32, device=dr.device), torch.exp(-(dr-torch.tensor(2.5, dtype=torch.float32, device=dr.device))**3))
    return f * m * expdmp

@partial(vmap, in_dims=(0, 0, 0, None, None, 0, None, None), out_dims=(0))
def disp_pme_real_kernel(dr, ci, cj, box, box_inv, mscales, kappa, pmax):
    '''
    The kernel to calculate the realspace dispersion energy
    
    Inputs:
        ri: 
            Np * 3: position i
        rj:
            Np * 3: position j
        ci: 
            Np * (pmax-4)/2: dispersion coeffs of i, c6, c8, c10 etc
        cj:
            Np * (pmax-4)/2: dispersion coeffs of j, c6, c8, c10 etc
        kappa:
            float: kappa
        pmax:
            int: largest p in 1/r^p, assume starting from 6 with increment of 2

    Output:
        energy: 
            float: the dispersion pme energy
    '''

    @jit_condition()
    def calc_e(dr, ci, cj, box, box_inv, mscales, kappa, pmax):
        dr2 = dr * dr
        #dr2 = torch.matmul(dr, dr)
        x2 = kappa * kappa * dr2
        g = g_p(x2, pmax)
        dr6 = dr2 * dr2 * dr2
        ene = (mscales + g[0] - 1) * ci[0] * cj[0] / dr6
        dr8 = dr6 * dr2; dr10 = dr8 * dr2
        ene8 = torch.where(pmax >= torch.tensor(8, dtype=torch.int32, device=mscales.device), (mscales + g[1] - 1) * ci[1] * cj[1] / dr8, torch.zeros_like(ene))
        ene10 = torch.where(pmax >= torch.tensor(10, dtype=torch.int32, device=mscales.device), (mscales + g[2] - 1) * ci[2] * cj[2] / dr10, torch.zeros_like(ene))
        ene = ene + ene8 + ene10
        #expdmp = torch.where(dr < torch.tensor(2.5, dtype=torch.float32, device=dr.device), torch.tensor(0., dtype=torch.float32, device=dr.device), torch.exp(-(dr-torch.tensor(2.5, dtype=torch.float32, device=dr.device))**3))
        expdmp = torch.tensor(1., dtype=torch.float32, device=dr.device)
        ene = ene * expdmp
        return ene
    ene = calc_e(dr, ci, cj, box, box_inv, mscales, kappa, pmax)
    return ene

@jit_condition()
def disp_pme_self(c_list, kappa, pmax):
    '''
    This function calculates the dispersion self energy

    Inputs:
        c_list:
            Na * 3: dispersion susceptibilities C_6, C_8, C_10
        kappa:
            float: kappa used in dispersion

    Output:
        ene_self:
            float: the self energy
    '''
    E_6 = -kappa**6/12 * torch.sum(c_list[:, 0]**2)
    E_8 = torch.where(pmax >= torch.tensor(8, dtype=torch.int32, device=E_6.device), -kappa**8/48 * torch.sum(c_list[:, 1]**2), torch.zeros_like(E_6))
    E_10 = torch.where(pmax >= torch.tensor(10, dtype=torch.int32, device=E_6.device), -kappa**10/240 * torch.sum(c_list[:, 2]**2), torch.zeros_like(E_6))
    E_6 = E_6 + E_8 + E_10
    return E_6


if __name__ == '__main__':
    # first we define the cov_map from the topo infor
    data = np.load('100K_properties.npz',allow_pickle=True)
    num_idx = 26
    positions = data['coord'][num_idx]; bonds = data['topo'][num_idx]
    c6 = data['c6'][num_idx]; c8 = data['c8'][num_idx]; c10 = data['c10'][num_idx]
    atoms = {'positions':positions,'bonds':bonds,'c6':c6,'c8':c8,'c10':c10}
    cov_map = build_covalent_map(atoms, 6)
    pair_full = []
    for na in range(len(atoms['positions'])):
        for nb in range(na + 1, len(atoms['positions'])):
            pair_full.append([na, nb, 0])
    pair_full = np.array(pair_full, dtype=int)
    pair_full[:,2] = cov_map[pair_full[:,0], pair_full[:,1]]

    pairs = torch.tensor(pair_full,requires_grad=False)
    mscales = torch.tensor([0., 0., 0., 0., 1., 1.], dtype=torch.float32, requires_grad=False)
    box = torch.tensor([[50.,0.,0.],[0.,50.,0.],[0.,0.,50.]], dtype=torch.float32, requires_grad=False)

    positions = torch.tensor(positions, dtype=torch.float32, requires_grad=True)
    c6 = torch.tensor(c6, dtype=torch.float32, requires_grad=False)
    c8 = torch.tensor(c8, dtype=torch.float32, requires_grad=False)
    c10 = torch.tensor(c10, dtype=torch.float32, requires_grad=False)
    c_list = torch.vstack((c6,c8,c10)).T

    e = energy_disp_pme(positions, box, pairs, c_list, mscales, 
                        None, None, None, None, 10, None, None, None, lpme=False)
    grad = torch.autograd.grad(outputs=e,inputs=positions)



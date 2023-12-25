#!/usr/bin/env python
import torch 
import functorch
from functorch import vmap
from dmff_torch.pairwise import distribute_scalar, distribute_multipoles, distribute_v3
from dmff_torch.utils import jit_condition  
from dmff_torch.spatial import generate_construct_local_frames
from dmff_torch.multipole import C1_c2h, rot_local2global 
from torch import erf, erfc
from dmff_torch.constants import DIELECTRIC
from dmff_torch.nblist import build_covalent_map
from functools import partial
import numpy as np
from dmff_torch.recip import pme_recip
from typing import Tuple, Optional
from dmff_torch.settings import POL_CONV, MAX_N_POL

# we use torch.autograd.grad to calculate the grad 
DEFAULT_THOLE_WIDTH = 5.0


class ADMPPmeForce:
    '''
    This is a convenient wrapper for multipolar PME calculations
    It wrapps all the environment parameters of multipolar PME calculation
    The so called "environment paramters" means parameters that do not need to be differentiable
    '''

    def __init__(self, box, axis_type, axis_indices, rc, ethresh, lmax, lpol=False, lpme=True, steps_pol=None):
        '''
        Initialize the ADMPPmeForce calculator.

        Input:
            box: 
                (3, 3) float, box size in row
            axis_type:
                (na,) int, types of local axis (bisector, z-then-x etc.)
            rc: 
                float: cutoff distance
            ethresh: 
                float: pme energy threshold
            lmax:
                int: max L for multipoles
            lpol:
                bool: polarize or not?
            lpme:
                bool: do pme or simple cutoff? 
                if False, the kappa will be set to zero and the reciprocal part will not be computed
            steps:
                None or int: Whether do fixed number of dipole iteration steps?
                if None: converge dipoles until convergence threshold is met
                if int: optimize for this many steps and stop, this is useful if you want to jit the entire function

        Output:

        '''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if axis_indices == None:
            self.axis_type = None
            self.axis_indices = None
        else:
            self.axis_type = axis_type
            self.axis_indices = axis_indices
        self.rc = rc
        self.ethresh = ethresh
        self.lmax = lmax  # jichen: type checking
        # turn off pme if lpme is False, this is useful when doing cluster calculations
        self.lpme = lpme
        if self.lpme is False:
            self.kappa = torch.tensor(0., dtype=torch.float32, device=self.device)
            self.K1 = torch.tensor(0., dtype=torch.float32, device=self.device)
            self.K2 = torch.tensor(0., dtype=torch.float32, device=self.device)
            self.K3 = torch.tensor(0., dtype=torch.float32, device=self.device)
            self.n_mesh = None
            self.shifts = None
        else:
            kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
            self.kappa = kappa
            self.K1 = K1
            self.K2 = K2
            self.K3 = K3
            #################################################################
            # modify here, for the torch.jit purpose
            pme_order = 6
            bspline_range = torch.arange(-pme_order//2, pme_order//2, dtype=torch.float32, device=self.device)
            n_mesh = pme_order**3
            shift_y,shift_x,shift_z = torch.meshgrid(bspline_range, bspline_range, bspline_range,indexing='ij')
            shifts = torch.stack((shift_x,shift_y,shift_z)).transpose(0,3).reshape((1,n_mesh,3))
            self.n_mesh = torch.tensor(n_mesh, dtype=torch.int32, device=self.device)
            self.shifts = shifts.to(torch.float32)
            ################################################################

        self.pme_order = 6
        self.lpol = lpol
        self.steps_pol = steps_pol
        self.n_atoms = len(axis_type)

        # setup calculators
        self.refresh_calculators()
        return

    def generate_energy(self, positions, box, pairs,
                    Q_local, pol, tholes, mScales, pScales, dScales, B, ldmp):
        # if the force field is not polarizable
        if not self.lpol:
            return energy_pme(positions, box, pairs,
                                 Q_local, None, None, None,
                                 mScales, None, None,
                                 self.construct_local_frames,
                                 self.kappa, self.K1, self.K2, self.K3, self.lmax, False, self.n_mesh, self.shifts, B, ldmp, lpme=self.lpme)
        else:
            U_init = torch.zeros((self.n_atoms, 3), dtype=torch.float32, device=positions.device, requires_grad=True)
            # this is the wrapper that include a Uind optimize    

            self.U_ind, lconverg, n_cycle = self.optimize_Uind(
                        positions, box, pairs, Q_local, pol, tholes,
                        mScales, pScales, dScales,
                        U_init=U_init, steps_pol=self.steps_pol)
                # here we rely on Feynman-Hellman theorem, drop the term dV/dU*dU/dr !
                # self.U_ind = jax.lax.stop_gradient(U_ind)
            return energy_pme(positions, box, pairs,
                                 Q_local, self.U_ind, pol, tholes,
                                 mScales, pScales, dScales,
                                 self.construct_local_frames, 
                                 self.kappa, self.K1, self.K2, self.K3, self.lmax, True, self.n_mesh, self.shifts, B, ldmp, lpme=self.lpme) 

    def update_env(self, attr, val):
        '''
        Update the environment of the calculator
        '''
        setattr(self, attr, val)
        self.refresh_calculators()

    def refresh_calculators(self):
        '''
        refresh the energy and force calculators according to the current environment
        '''
        if self.lmax > 0:
            if self.axis_type == None:
                self.construct_local_frames = None
            else:
                self.construct_local_frames = generate_construct_local_frames(self.axis_type, self.axis_indices)
        else:
            self.construct_local_frames = None
        
        lmax = self.lmax
        # for polarizable monopole force field, need to increase lmax to 1, accomodating induced dipoles
        if self.lmax == 0 and self.lpol is True:
            lmax = 1
        if self.lpol:
            self.grad_U_fn = functorch.grad(energy_pme, argnums=(4))
        self.get_energy = self.generate_energy
        return

    def optimize_Uind(self, 
                positions, box, pairs,
                Q_local, pol, tholes, mScales, pScales, dScales,
                U_init=None, steps_pol=None):
        '''
        This function converges the induced dipole
        Note that we cut all the gradient chain passing through this function as we assume Feynman-Hellman theorem
        Gradients related to Uind should be dropped
        '''
        # Do not track gradient in Uind optimization
        maxiter = 30 
        thresh = 1.0
        positions = positions.detach()
        box = box.detach()
        Q_local = Q_local.detach()
        pol = pol.detach()
        tholes = tholes.detach()
        mScales = mScales.detach()
        pScales = pScales.detach()
        dScales = dScales.detach()
        if U_init is None:
            U = torch.zeros_like(positions, dtype=torch.float32, device=positions.device, requires_grad=True)
        else:
            U = U_init
        if steps_pol is None:
            site_filter = (pol>0.001) # focus on the actual polarizable sites
    
        if steps_pol is None:
            for i in range(maxiter):
                field = self.grad_U_fn(positions, box, pairs, Q_local, U, pol, tholes, mScales, pScales, dScales, self.construct_local_frames, self.kappa, self.K1, self.K2, self.K3, self.lmax, True, self.n_mesh, self.shifts, None, False, lpme=self.lpme)
                if torch.max(torch.abs(field[site_filter])) < thresh:
                    break
                U = U - field * pol[:, None] / DIELECTRIC
            if i == maxiter-1:
                flag = False
            else: # converged
                flag = True
        else:
            def update_U(i, U):
                field = self.grad_U_fn(positions, box, pairs, Q_local, U, pol, tholes, mScales, pScales, dScales, self.construct_local_frames, self.kappa, self.K1, self.K2, self.K3, self.lmax, True, self.n_mesh, self.shifts, None, False, lpme=self.lpme)
                U = U - field * pol[:, None] / DIELECTRIC
                return U
            # here check the U_ind
            #e = self.energy_fn(positions, box, pairs, Q_local, Uind_global, pol, tholes, mScales, pScales, dScales)
            #field = self.grad_U_fn(positions, box, pairs, Q_local, Uind_global, pol, tholes, mScales, pScales, dScales)
            for ii in range(0, steps_pol):
                U = update_U(ii, U)
            #U = jax.lax.fori_loop(0, steps_pol, update_U, U)
            flag = True
        return U, flag, steps_pol

def pme_real(positions, box, pairs, 
        Q_global, Uind_global, pol, tholes, 
        mScales, pScales, dScales, 
        kappa, lmax, lpol, B, ldmp):
    '''
    This is the real space PME calculate function 
    NOTE: only deals with permanent-permanent multipole interactions
    for jax, it is pointless to jit it 
    Input:
       positions:
           Na * 3: positions
       box: 
           3 * 3: box, axes arranged in row 
       pairs: 
           Np * 3: interacting pair indices and topology distance
       Q_global:
           Na * (l+1)**2: harmonics multipoles of each atom, in global frame
       Uind_global:
           Na * 3: harmonic induced dipoles, in global frame
       pol:
          (Na,): polarizabilities
       tholes:
          (Na,): thole damping parameters
       mScales:
          (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
       covalent_map:
          Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        lmax:
            int: maximum L
        lpol:
            Bool: whether do a polarizable calculation?
    Output:
        ene: pme realspace energy 
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
 
    pairs[:,:2] = regularize_pairs(pairs[:,:2])
    buffer_scales = pair_buffer_scales(pairs[:, :2])
    box_inv = torch.linalg.inv(box)

    
    r1 = distribute_v3(positions.T, pairs[:,0]).T
    r2 = distribute_v3(positions.T, pairs[:,1]).T

    Q_extendi = distribute_multipoles(Q_global.T, pairs[:, 0]).T
    Q_extendj = distribute_multipoles(Q_global.T, pairs[:, 1]).T
    
    nbonds = pairs[:,2]
    indices = (nbonds + (mScales.shape[0] - 1)) % mScales.shape[0]
    
    mscales = distribute_scalar(mScales, indices)
    mscales = mscales * buffer_scales

    @partial(vmap, in_dims=(0, 0), out_dims=(0))
    @jit_condition()
    def get_pair_dmp(pol1, pol2):
        return (pol1*pol2) ** (1/6)
    
    if ldmp:
        B_i = distribute_scalar(B, pairs[:,0])
        B_j = distribute_scalar(B, pairs[:,1])

    if lpol:
        pol1 = distribute_scalar(pol, pairs[:, 0])
        pol2 = distribute_scalar(pol, pairs[:, 1])
        thole1 = distribute_scalar(tholes, pairs[:, 0])
        thole2 = distribute_scalar(tholes, pairs[:, 1])
        Uind_extendi = distribute_v3(Uind_global.T, pairs[:, 0]).T
        Uind_extendj = distribute_v3(Uind_global.T, pairs[:, 1]).T
        pscales = distribute_scalar(pScales, indices)
        pscales = pscales * buffer_scales
        dscales = distribute_scalar(dScales, indices)
        dscales = dscales * buffer_scales
        dmp = get_pair_dmp(pol1, pol2)
    else:
        Uind_extendi = None
        Uind_extendj = None
        pscales = None
        dscales = None
        thole1 = None
        thole2 = None
        dmp = None
    
    @partial(vmap, in_dims=(0, None, None), out_dims=0)
    @jit_condition()
    def v_pbc_shift(drvecs, box, box_inv):
        unshifted_dsvecs = torch.matmul(drvecs, box_inv)
        dsvecs = unshifted_dsvecs - torch.floor(unshifted_dsvecs + 0.5)
        return torch.matmul(dsvecs, box)

    @partial(vmap, in_dims=(0, 0, 0, 0), out_dims=0)
    @jit_condition()
    def build_quasi_internal(r1, r2, dr, norm_dr, bias_0 = torch.tensor([1., 0., 0.],dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu'), bias_1 = torch.tensor([0., 1., 0.], dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')):
        '''
        Build the quasi-internal frame between a pair of sites
        In this frame, the z-axis is pointing from r2 to r1

        Input:
            r1:
                N * 3, positions of the first vector
            r2:
                N * 3, positions of the second vector
            dr:
                N * 3, vector pointing from r1 to r2
            norm_dr:
                (N,), distances between r1 and r2
 
        Output:
            local_frames:
                N * 3 * 3: local frames, three axes arranged in rows
        '''
        vectorZ = dr/norm_dr
        vectorX = torch.where(torch.logical_or(r1[1]!=r2[1],r1[2]!=r2[2]),vectorZ + bias_0, vectorZ + bias_1)

        dot_xz = torch.matmul(vectorZ, vectorX)
        vectorX = vectorX - vectorZ * dot_xz
        vectorX = vectorX / torch.norm(vectorX)
        vectorY = torch.cross(vectorZ,vectorX)
        return torch.stack([vectorX, vectorY, vectorZ])

    @partial(vmap, in_dims=(0, 0), out_dims=0)
    @jit_condition()
    def rot_ind_global2local(U_g, localframes, zxy = torch.tensor([2,0,1], dtype=torch.long, device='cuda:0' if torch.cuda.is_available() else 'cpu')):
        '''
        A special rotation function for just dipoles, aim for applying on induced dipoles
        '''
        R1 = localframes[zxy][:,zxy]
        U_l = torch.matmul(R1, U_g)
        return U_l

    @partial(vmap, in_dims=(0, 0, 0, 0, 0, 0, 0, None), out_dims=0)
    @jit_condition()
    def rot_global2local(Qi_0, Qi_1, Qi_2, Qj_0, Qj_1, Qj_2, localframes, lmax, rt3=torch.tensor(1.73205080757,dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu'), zxy=torch.tensor([2,0,1], dtype=torch.long, device='cuda:0' if torch.cuda.is_available() else 'cpu')):
        '''
        This function rotates harmonic moments Q from global frame to local frame

        Input:
            Q_gh:
                n * (l+1)^2, stores the global harmonic multipole moments of each site
            localframes:
                n * 3 * 3, stores the Rotation matrix for each site, the R is defined as:
                [r1, r2, r3]^T, with r1, r2, r3 being the local frame axes
            lmax:
                integer, the maximum multipole order
            C2_gl: the local rotate matrix when lmax=2
            zxy: the harmonic transform matrix when lmax=1

        Output:
            Q_lh:
                n * (l+1)^2, stores the local harmonic multipole moments
        '''
        #rt3 = 1.73205080757
        inv_rt3 = 1.0/rt3

        # monopole
        if lmax < 1:
            Qi_lh = Qi_0
            Qj_lh = Qj_0
        # dipole 
        elif lmax < 2:
            # the rotation matrix
            #zxy = [2,0,1]
            R1 = localframes[zxy][:,zxy]
            # rotate
            Qi_lh_1 = torch.matmul(R1, Qi_1)
            Qj_lh_1 = torch.matmul(R1, Qj_1)
            Qi_lh = torch.hstack([Qi_0, Qi_lh_1])
            Qj_lh = torch.hstack([Qj_0, Qj_lh_1])
        else:
            # the rotation matrix
            #zxy = [2,0,1]
            R1 = localframes[zxy][:,zxy]
            # rotate
            Qi_lh_1 = torch.matmul(R1, Qi_1)
            Qj_lh_1 = torch.matmul(R1, Qj_1)
            xx = localframes[0, 0]
            xy = localframes[0, 1]
            xz = localframes[0, 2]
            yx = localframes[1, 0]
            yy = localframes[1, 1]
            yz = localframes[1, 2]
            zx = localframes[2, 0]
            zy = localframes[2, 1]
            zz = localframes[2, 2]
            # construct the local->global transformation matrix
            # this is copied directly from the convert_mom_to_xml.py code
            C2_gl_00 = (3*zz**2-1)/2
            C2_gl_01 = rt3*zx*zz
            C2_gl_02 = rt3*zy*zz
            C2_gl_03 = (rt3*(-2*zy**2-zz**2+1))/2
            C2_gl_04 = rt3*zx*zy
            C2_gl_10 = rt3*xz*zz
            C2_gl_11 = 2*xx*zz-yy
            C2_gl_12 = yx+2*xy*zz
            C2_gl_13 = -2*xy*zy-xz*zz
            C2_gl_14 = xx*zy+zx*xy
            C2_gl_20 = rt3*yz*zz
            C2_gl_21 = 2*yx*zz+xy
            C2_gl_22 = -xx+2*yy*zz
            C2_gl_23 = -2*yy*zy-yz*zz
            C2_gl_24 = yx*zy+zx*yy
            C2_gl_30 = rt3*(-2*yz**2-zz**2+1)/2
            C2_gl_31 = -2*yx*yz-zx*zz
            C2_gl_32 = -2*yy*yz-zy*zz
            C2_gl_33 = (4*yy**2+2*zy**2+2*yz**2+zz**2-3)/2
            C2_gl_34 = -2*yx*yy-zx*zy
            C2_gl_40 = rt3*xz*yz
            C2_gl_41 = xx*yz+yx*xz
            C2_gl_42 = xy*yz+yy*xz
            C2_gl_43 = -2*xy*yy-xz*yz
            C2_gl_44 = xx*yy+yx*xy
            C2_gl = torch.stack(
                (
                    torch.stack((C2_gl_00, C2_gl_10, C2_gl_20, C2_gl_30, C2_gl_40)),
                    torch.stack((C2_gl_01, C2_gl_11, C2_gl_21, C2_gl_31, C2_gl_41)),
                    torch.stack((C2_gl_02, C2_gl_12, C2_gl_22, C2_gl_32, C2_gl_42)),
                    torch.stack((C2_gl_03, C2_gl_13, C2_gl_23, C2_gl_33, C2_gl_43)),
                    torch.stack((C2_gl_04, C2_gl_14, C2_gl_24, C2_gl_34, C2_gl_44))
                )
            )

            C2_gl = torch.transpose(C2_gl,0,1)
            Qi_lh_2 = torch.einsum('jk,k->j', C2_gl, Qi_2)
            Qj_lh_2 = torch.einsum('jk,k->j', C2_gl, Qj_2)
            Qi_lh = torch.hstack([Qi_0, Qi_lh_1, Qi_lh_2]) 
            Qj_lh = torch.hstack([Qj_0, Qj_lh_1, Qj_lh_2])
        return Qi_lh, Qj_lh

    # deals with geometries
    dr = r1 - r2
    dr = v_pbc_shift(dr, box, box_inv)
    norm_dr = torch.linalg.norm(dr, dim=-1)
    Ri = build_quasi_internal(r1, r2, dr, norm_dr.unsqueeze(-1))
    ########################################################################
    Q_0i = Q_extendi[:,0:1]; Q_1i = Q_extendi[:,1:4]; Q_2i = Q_extendi[:,4:9]
    Q_0j = Q_extendj[:,0:1]; Q_1j = Q_extendj[:,1:4]; Q_2j = Q_extendj[:,4:9]
    qiQI, qiQJ = rot_global2local(Q_0i, Q_1i, Q_2i, Q_0j, Q_1j, Q_2j, Ri, lmax)
    #qiQJ = rot_global2local(Q_0j, Q_1j, Q_2j, Ri, lmax)

    ########################################################################
    if lpol:
        qiUindI = rot_ind_global2local(Uind_extendi, Ri)
        qiUindJ = rot_ind_global2local(Uind_extendj, Ri)
    else:
        qiUindI = None
        qiUindJ = None
    
    # everything should be pair-specific now
    if lpol:    
        if ldmp :
            ene = torch.sum(
              pme_dmp_kernel(
              norm_dr,
              Q_0i, 
              Q_0j,
              B_i,
              B_j,
              mscales) * buffer_scales) + torch.sum( 
              pme_real_kernel_pol(
              norm_dr,
              qiQI,
              qiQJ,
              qiUindI,
              qiUindJ,
              thole1,
              thole2,
              dmp,
              mscales,
              pscales,
              dscales,
              kappa,
              lmax,
              lpol
              ) * buffer_scales)
        else:
            ene = torch.sum(
              pme_real_kernel_pol(
              norm_dr, 
              qiQI,
              qiQJ,
              qiUindI,
              qiUindJ,
              thole1,
              thole2,
              dmp,
              mscales,
              pscales,
              dscales,
              kappa,
              lmax,
              lpol
         ) * buffer_scales
    )
    else:
        ene = torch.sum(
          pme_real_kernel_nopol(
              norm_dr,
              qiQI,
              qiQJ,
              qiUindI,
              qiUindJ,
              thole1,
              thole2,
              dmp,
              mscales,
              pscales,
              dscales,
              kappa,
              lmax,
              lpol
         ) * buffer_scales
    )
    return ene 


@vmap
#@jit_condition()
def pme_dmp_kernel(dr, Qi, Qj, Bi, Bj, m):
    DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device=dr.device)
    b = torch.sqrt(Bi * Bj); q = Qi[0] * Qj[0]
    br = b * dr
    exp_br = torch.exp(-br)
    #expdmp = torch.where(dr < torch.tensor(2.5, dtype=torch.float32, device=dr.device), torch.tensor(0., dtype=torch.float32, device=dr.device), torch.exp(-(dr-torch.tensor(2.5, dtype=torch.float32, device=dr.device))**3))
    e_tot = - exp_br * (1 + br) * q / dr * DIELECTRIC #* expdmp
    return e_tot * m 

@partial(vmap, in_dims=(0, 0, 0, None, None, None, None, None, 0, None, None, None, None, None), out_dims=0)
def pme_real_kernel_nopol(dr, qiQI, qiQJ, qiUindI, qiUindJ, thole1, thole2, dmp, mscales, pscales, dscales, kappa, lmax, lpol=False):
    '''
    This is the heavy-lifting kernel function to compute the realspace multipolar PME 
    Vectorized over interacting pairs

    Input:
        dr: 
            float, the interatomic distances, (np) array if vectorized
        qiQI:
            [(lmax+1)^2] float array, the harmonic multipoles of site i in quasi-internal frame
        qiQJ:
            [(lmax+1)^2] float array, the harmonic multipoles of site j in quasi-internal frame
        qiUindI
            (3,) float array, the harmonic dipoles of site i in QI frame
        qiUindJ
            (3,) float array, the harmonic dipoles of site j in QI frame
        thole1
            float: thole damping coeff of site i
        thole2
            float: thole damping coeff of site j
        dmp:
            float: (pol1 * pol2)**1/6, distance rescaling params used in thole damping
        mscale:
            float, scaling factor between interacting sites (permanent-permanent)
        pscale:
            float, scaling factor between perm-ind interaction
        dscale:
            float, scaling factor between ind-ind interaction
        kappa:
            float, kappa in unit A^1
        lmax:
            int, maximum angular momentum
        lpol:
            bool, doing polarization?

    Output:
        energy: 
    '''
    # as we using vmap, thus, the tensor is from 2D to 1D
    @jit_condition()
    def calc_e_perm(dr, mscales, kappa, lmax):
                        
        r'''
        This function calculates the ePermCoefs at once
        ePermCoefs is basically the interaction tensor between permanent multipole components
        Everything should be done in the so called quasi-internal (qi) frame
        Energy = \sum_ij qiQI * ePermCoeff_ij * qiQJ

        Inputs:
            dr: 
                float: distance between one pair of particles
            mscales:
                float: scaling factor between permanent - permanent multipole interactions, for each pair
            kappa:
                float: \kappa in PME, unit in A^-1
            lmax:
                int: max L

        Output:
            cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2:
                n * 1 array: ePermCoefs
        '''
        # torch.Tensor can not combine with vmap
        DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device=dr.device)

        rInv = 1 / dr
        
        rInvVec = [DIELECTRIC*(rInv**i) for i in range(0, 9)]

        alphaRVec = [(kappa*dr)**i for i in range(0, 10)]
        
        X = 2 * torch.exp(-alphaRVec[2]) / torch.sqrt(torch.tensor(torch.pi, dtype=torch.float32, device=dr.device))
        tmp = alphaRVec[1]
        doubleFactorial = 1
        facCount = 1
        erfAlphaR = erf(alphaRVec[1])
        
        # the calc_kernel using the pmap, dr is a value
        bVec = [torch.zeros_like(erfAlphaR), -erfAlphaR]

        #bVec = torch.empty(6)
        
        #bVec[1] = -erfAlphaR
        for i in range(2, 6):
            bVec.append(bVec[i-1] + (tmp*X/doubleFactorial))
            facCount = facCount + 2
            doubleFactorial = doubleFactorial * facCount
            tmp = tmp * 2 * alphaRVec[2]

        # in pme we need add erfc function in 1/rij 
        cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1]*X)

        if lmax >= 1:
            # C-D
            cd = rInvVec[2] * (mscales + bVec[2])
            # D-D: 2
            dd_m0 = -2/3 * rInvVec[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X)
            dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)
        else:
            cd = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dd_m0 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dd_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device)

        if lmax >= 2:
            ## C-Q: 1
            cq = (mscales + bVec[3]) * rInvVec[3]
            ## D-Q: 2
            dq_m0 = rInvVec[4] * (3* (mscales + bVec[3]) + (4/3) * alphaRVec[5]*X)
            dq_m1 = -torch.sqrt(torch.tensor(3., dtype=torch.float32, device=dr.device)) * rInvVec[4] * (mscales + bVec[3])
            ## Q-Q
            qq_m0 = rInvVec[5] * (6* (mscales + bVec[4]) + (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
            qq_m1 = - (4/15) * rInvVec[5] * (15*(mscales+bVec[4]) + alphaRVec[5]*X)
            qq_m2 = rInvVec[5] * (mscales + bVec[4] - (4/15)*alphaRVec[5]*X)
        else:
            cq = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dq_m0 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dq_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device) 
            qq_m0 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            qq_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            qq_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            qq_m2 = torch.tensor(0., dtype=torch.float32, device=dr.device)
        ######################################################################
        # add the damping function here !!!!
        ######################################################################
        #expdmp = torch.where(dr < torch.tensor(2.5, dtype=torch.float32, device=dr.device), torch.tensor(0., dtype=torch.float32, device=dr.device), torch.exp(-(dr-torch.tensor(2.5, dtype=torch.float32, device=dr.device))**3))
        expdmp = torch.tensor(1., dtype=torch.float32, device=dr.device)
        return cc*expdmp, cd*expdmp, dd_m0*expdmp, dd_m1*expdmp, cq*expdmp, dq_m0*expdmp, dq_m1*expdmp, qq_m0*expdmp, qq_m1*expdmp, qq_m2*expdmp

    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_e_perm(dr, mscales, kappa, lmax)

    @jit_condition()
    def trim_val_0(x):
        return torch.where(x < torch.tensor(1e-8, dtype=torch.float32, device=x.device), torch.tensor(1e-8, dtype=torch.float32, device=x.device), x)
    
    @jit_condition()
    def trim_val_infty(x):
        return torch.where(x > torch.tensor(1e8, dtype=torch.float32, device=x.device), torch.tensor(1e8, dtype=torch.float32, device=x.device), x)
    
    @jit_condition()
    def calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax):

        '''
        This function calculates the eUindCoefs at once
           ## compute the Thole damping factors for energies
         eUindCoefs is basically the interaction tensor between permanent multipole components and induced dipoles
        Everything should be done in the so called quasi-internal (qi) frame
        

        Inputs:
            dr: 
                float: distance between one pair of particles
            dmp
                float: damping factors between one pair of particles
            mscales:
                float: scaling factor between permanent - permanent multipole interactions, for each pair
            pscales:
                float: scaling factor between permanent - induced multipole interactions, for each pair
            au:
                float: for damping factors
            kappa:
                float: \kappa in PME, unit in A^-1
            lmax:
                int: max L

        Output:
            Interaction tensors components
        '''
        DEFAULT_THOLE_WIDTH = 5.0
        ## pscale == 0 ? thole1 + thole2 : DEFAULT_THOLE_WIDTH`
        w = torch.heaviside(pscales, torch.tensor(0., dtype=torch.float32, device=dr.device))
        a = w * DEFAULT_THOLE_WIDTH + (1-w) * (thole1+thole2)
        dmp = trim_val_0(dmp)
        u = trim_val_infty(dr/dmp)

        ## au <= 50 aupi = au ;au> 50 aupi = 50
        au = a * u
        expau = torch.where(au < torch.tensor(50.0, dtype=torch.float32, device=dr.device), torch.exp(-au), torch.tensor(0.,dtype=torch.float32, device=dr.device))

        ## compute the Thole damping factors for energies
        au2 = trim_val_infty(au*au)
        au3 = trim_val_infty(au2*au)
        au4 = trim_val_infty(au3*au)
        au5 = trim_val_infty(au4*au)
        au6 = trim_val_infty(au5*au)

        ##  Thole damping factors for energies
        thole_c   = 1.0 - expau*(1.0 + au + 0.5*au2)
        thole_d0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0)
        thole_d1  = 1.0 - expau*(1.0 + au + 0.5*au2)
        thole_q0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/18.0)
        thole_q1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0)

        rInv = 1. / dr

        rInvVec = [DIELECTRIC*(rInv**i) for i in range(0, 9)]

        alphaRVec = [(kappa*dr)**i for i in range(0, 10)]

        X = 2. * torch.exp(-alphaRVec[2]) / torch.sqrt(torch.tensor(torch.pi))
        tmp = alphaRVec[1]
        doubleFactorial = 1
        facCount = 1
        erfAlphaR = erf(alphaRVec[1])

        # the calc_kernel using the pmap, dr is a value
        bVec = [torch.zeros_like(erfAlphaR), -erfAlphaR]

        #bVec = torch.empty(6)

        #bVec[1] = -erfAlphaR
        for i in range(2, 6):
            bVec.append(bVec[i-1] + (tmp*X/doubleFactorial))
            facCount = facCount + 2.0
            doubleFactorial = doubleFactorial * facCount
            tmp = tmp * 2.0 * alphaRVec[2]

        ## C-Uind
        cud = 2.0*rInvVec[2]*(pscales*thole_c + bVec[2])
        if lmax >= 1:
            ##  D-Uind terms
            dud_m0 = -2.0*2.0/3.0*rInvVec[3]*(3.0*(pscales*thole_d0 + bVec[3]) + alphaRVec[3]*X)
            dud_m1 = 2.0*rInvVec[3]*(pscales*thole_d1 + bVec[3] - 2.0/3.0*alphaRVec[3]*X)
        else:
            dud_m0 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)
            dud_m1 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)

        if lmax >= 2:
            udq_m0 = 2.0*rInvVec[4]*(3.0*(pscales*thole_q0 + bVec[3]) + 4/3*alphaRVec[5]*X)
            udq_m1 =  -2.0*torch.sqrt(torch.tensor(3))*rInvVec[4]*(pscales*thole_q1 + bVec[3])
        else:
            udq_m0 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)
            udq_m1 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)
        ## Uind-Uind
        udud_m0 = -2.0/3.0*rInvVec[3]*(3.0*(dscales*thole_d0 + bVec[3]) + alphaRVec[3]*X)
        udud_m1 = rInvVec[3]*(dscales*thole_d1 + bVec[3] - 2.0/3.0*alphaRVec[3]*X)
        
        #expdmp = torch.where(dr < torch.tensor(2.5, dtype=torch.float32, device=dr.device), torch.tensor(0., dtype=torch.float32, device=dr.device), torch.exp(-(dr-torch.tensor(2.5, dtype=torch.float32, device=dr.device))**3))
        return cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1

    if lpol:
        cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1 = calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax)
    
    @jit_condition()
    def calc_e_tot(qiQI, qiQJ, qiUindI, qiUindJ, lmax, lpol, cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1):
        Vij0 = cc*qiQI[0]
        Vji0 = cc*qiQJ[0]
        # C-Uind
        if lpol > 0:
            Vij0 -= cud * qiUindI[0]
            Vji0 += cud * qiUindJ[0]

        if lmax >= 1:
            # C-D
            Vij0 = Vij0 - cd*qiQI[1]
            Vji1 = -cd*qiQJ[0]
            Vij1 = cd*qiQI[0]
            Vji0 = Vji0 + cd*qiQJ[1]
            # D-D m0
            Vij1 += dd_m0 * qiQI[1]
            Vji1 += dd_m0 * qiQJ[1]
            # D-D m1
            Vij2 = dd_m1*qiQI[2]
            Vji2 = dd_m1*qiQJ[2]
            Vij3 = dd_m1*qiQI[3]
            Vji3 = dd_m1*qiQJ[3]
            # D-Uind
            if lpol > 0:
                Vij1 += dud_m0 * qiUindI[0]
                Vji1 += dud_m0 * qiUindJ[0]
                Vij2 += dud_m1 * qiUindI[1]
                Vji2 += dud_m1 * qiUindJ[1]
                Vij3 += dud_m1 * qiUindI[2]
                Vji3 += dud_m1 * qiUindJ[2]
        else:
            Vij0 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji1 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij1 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            # D-D m1
            Vij2 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji2 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij3 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji3 = torch.tensor(0., dtype=torch.float32, device=cc.device)

        if lmax >= 2:
            # C-Q
            Vij0 = Vij0 + cq*qiQI[4]
            Vji4 = cq*qiQJ[0]
            Vij4 = cq*qiQI[0]
            Vji0 = Vji0 + cq*qiQJ[4]
            # D-Q m0 
            Vij1 += dq_m0*qiQI[4]
            Vji4 += dq_m0*qiQJ[1]
            # Q-D m0 
            Vij4 -= dq_m0*qiQI[1]
            Vji1 -= dq_m0*qiQJ[4]
            # D-Q m1
            Vij2 = Vij2 + dq_m1*qiQI[5]
            Vji5 = dq_m1*qiQJ[2]
            Vij3 += dq_m1*qiQI[6]
            Vji6 = dq_m1*qiQJ[3]
            Vij5 = -(dq_m1*qiQI[2])
            Vji2 += -(dq_m1*qiQJ[5])
            Vij6 = -(dq_m1*qiQI[3])
            Vji3 += -(dq_m1*qiQJ[6])
            # Q-Q m0
            Vij4 += qq_m0*qiQI[4]
            Vji4 += qq_m0*qiQJ[4]
            # Q-Q m1
            Vij5 += qq_m1*qiQI[5]
            Vji5 += qq_m1*qiQJ[5]
            Vij6 += qq_m1*qiQI[6]
            Vji6 += qq_m1*qiQJ[6]
            # Q-Q m2
            Vij7  = qq_m2*qiQI[7]
            Vji7  = qq_m2*qiQJ[7]
            Vij8  = qq_m2*qiQI[8]
            Vji8  = qq_m2*qiQJ[8]
            # Q-Uind
            if lpol > 0:
                Vji4 += udq_m0*qiUindJ[0]
                Vij4 -= udq_m0*qiUindI[0]
                Vji5 += udq_m1*qiUindJ[1]
                Vji6 += udq_m1*qiUindJ[2]
                Vij5 -= udq_m1*qiUindI[1]
                Vij6 -= udq_m1*qiUindI[2]
        else:
            # C-Q
            Vji4 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij4 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            # D-Q m1
            Vji5 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji6 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij5 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij6 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            # Q-Q m2
            Vij7  = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji7  = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij8  = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji8  = torch.tensor(0., dtype=torch.float32, device=cc.device)


        # Uind - Uind
        if lpol > 0:
            Vij1dd = udud_m0 * qiUindI[0]
            Vji1dd = udud_m0 * qiUindJ[0]
            Vij2dd = udud_m1 * qiUindI[1]
            Vji2dd = udud_m1 * qiUindJ[1]
            Vij3dd = udud_m1 * qiUindI[2]
            Vji3dd = udud_m1 * qiUindJ[2]
            Vijdd = torch.stack(( Vij1dd, Vij2dd, Vij3dd))
            Vjidd = torch.stack(( Vji1dd, Vji2dd, Vji3dd))
        else:
            Vij1dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji1dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij2dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji2dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij3dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji3dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vijdd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vjidd = torch.tensor(0., dtype=torch.float32, device=cc.device)


        if lmax == 0:
            Vij = Vij0
            Vji = Vji0
        elif lmax == 1:
            Vij = torch.stack((Vij0, Vij1, Vij2, Vij3))
            Vji = torch.stack((Vji0, Vji1, Vji2, Vji3))
        elif lmax == 2:
            Vij = torch.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
            Vji = torch.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))
        else:
            raise ValueError(f"Invalid lmax {lmax}. Valid values are 0, 1, 2")
        if lpol > 0:
            return 0.5 * (torch.sum(qiQJ*Vij) + torch.sum(qiQI*Vji)) + 0.5 * (torch.sum(qiUindJ*Vijdd) + torch.sum(qiUindI*Vjidd))
        else:
            return 0.5 * (torch.sum(qiQJ*Vij) + torch.sum(qiQI*Vji))
    if lpol == True:
        e_tot = calc_e_tot(qiQI, qiQJ, qiUindI, qiUindJ, lmax, torch.tensor(1., dtype=torch.float32, device=cc.device), cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1)
    else:
        e_tot = calc_e_tot(qiQI, qiQJ, qiUindI, qiUindJ, lmax, torch.tensor(0., dtype=torch.float32, device=cc.device), cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, None, None, None, None, None, None, None)
    return e_tot


@partial(vmap, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None), out_dims=0)
def pme_real_kernel_pol(dr, qiQI, qiQJ, qiUindI, qiUindJ, thole1, thole2, dmp, mscales, pscales, dscales, kappa, lmax, lpol=False):
    '''
    This is the heavy-lifting kernel function to compute the realspace multipolar PME 
    Vectorized over interacting pairs

    Input:
        dr: 
            float, the interatomic distances, (np) array if vectorized
        qiQI:
            [(lmax+1)^2] float array, the harmonic multipoles of site i in quasi-internal frame
        qiQJ:
            [(lmax+1)^2] float array, the harmonic multipoles of site j in quasi-internal frame
        qiUindI
            (3,) float array, the harmonic dipoles of site i in QI frame
        qiUindJ
            (3,) float array, the harmonic dipoles of site j in QI frame
        thole1
            float: thole damping coeff of site i
        thole2
            float: thole damping coeff of site j
        dmp:
            float: (pol1 * pol2)**1/6, distance rescaling params used in thole damping
        mscale:
            float, scaling factor between interacting sites (permanent-permanent)
        pscale:
            float, scaling factor between perm-ind interaction
        dscale:
            float, scaling factor between ind-ind interaction
        kappa:
            float, kappa in unit A^1
        lmax:
            int, maximum angular momentum
        lpol:
            bool, doing polarization?

    Output:
        energy: 
    '''
    # as we using vmap, thus, the tensor is from 2D to 1D
    @jit_condition()
    def calc_e_perm(dr, mscales, kappa, lmax):
                        
        r'''
        This function calculates the ePermCoefs at once
        ePermCoefs is basically the interaction tensor between permanent multipole components
        Everything should be done in the so called quasi-internal (qi) frame
        Energy = \sum_ij qiQI * ePermCoeff_ij * qiQJ

        Inputs:
            dr: 
                float: distance between one pair of particles
            mscales:
                float: scaling factor between permanent - permanent multipole interactions, for each pair
            kappa:
                float: \kappa in PME, unit in A^-1
            lmax:
                int: max L

        Output:
            cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2:
                n * 1 array: ePermCoefs
        '''
        # torch.Tensor can not combine with vmap
        DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device=dr.device)

        rInv = 1 / dr
        
        rInvVec = [DIELECTRIC*(rInv**i) for i in range(0, 9)]

        alphaRVec = [(kappa*dr)**i for i in range(0, 10)]
        
        X = 2 * torch.exp(-alphaRVec[2]) / torch.sqrt(torch.tensor(torch.pi, dtype=torch.float32, device=dr.device))
        tmp = alphaRVec[1]
        doubleFactorial = 1
        facCount = 1
        erfAlphaR = erf(alphaRVec[1])
        
        # the calc_kernel using the pmap, dr is a value
        bVec = [torch.zeros_like(erfAlphaR), -erfAlphaR]

        #bVec = torch.empty(6)
        
        #bVec[1] = -erfAlphaR
        for i in range(2, 6):
            bVec.append(bVec[i-1] + (tmp*X/doubleFactorial))
            facCount = facCount + 2
            doubleFactorial = doubleFactorial * facCount
            tmp = tmp * 2 * alphaRVec[2]

        # in pme we need add erfc function in 1/rij 
        cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1]*X)

        if lmax >= 1:
            # C-D
            cd = rInvVec[2] * (mscales + bVec[2])
            # D-D: 2
            dd_m0 = -2/3 * rInvVec[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X)
            dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)
        else:
            cd = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dd_m0 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dd_m1 = torch.tensor(0.,dtype=torch.float32, device=dr.device)

        if lmax >= 2:
            ## C-Q: 1
            cq = (mscales + bVec[3]) * rInvVec[3]
            ## D-Q: 2
            dq_m0 = rInvVec[4] * (3* (mscales + bVec[3]) + (4/3) * alphaRVec[5]*X)
            dq_m1 = -torch.sqrt(torch.tensor(3., dtype=torch.float32, device=dr.device)) * rInvVec[4] * (mscales + bVec[3])
            ## Q-Q
            qq_m0 = rInvVec[5] * (6* (mscales + bVec[4]) + (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
            qq_m1 = - (4/15) * rInvVec[5] * (15*(mscales+bVec[4]) + alphaRVec[5]*X)
            qq_m2 = rInvVec[5] * (mscales + bVec[4] - (4/15)*alphaRVec[5]*X)
        else:
            cq = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dq_m0 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            dq_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device) 
            qq_m0 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            qq_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            qq_m1 = torch.tensor(0., dtype=torch.float32, device=dr.device)
            qq_m2 = torch.tensor(0., dtype=torch.float32, device=dr.device)
        return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2
    
    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_e_perm(dr, mscales, kappa, lmax)
    
    @jit_condition()
    def trim_val_0(x):
        return torch.where(x < torch.tensor(1e-8, dtype=torch.float32, device=x.device), torch.tensor(1e-8, dtype=torch.float32, device=x.device), x)
    
    @jit_condition()
    def trim_val_infty(x):
        return torch.where(x > torch.tensor(1e8, dtype=torch.float32, device=x.device), torch.tensor(1e8, dtype=torch.float32, device=x.device), x)
    
    @jit_condition()
    def calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax):

        '''
        This function calculates the eUindCoefs at once
           ## compute the Thole damping factors for energies
         eUindCoefs is basically the interaction tensor between permanent multipole components and induced dipoles
        Everything should be done in the so called quasi-internal (qi) frame
        

        Inputs:
            dr: 
                float: distance between one pair of particles
            dmp
                float: damping factors between one pair of particles
            mscales:
                float: scaling factor between permanent - permanent multipole interactions, for each pair
            pscales:
                float: scaling factor between permanent - induced multipole interactions, for each pair
            au:
                float: for damping factors
            kappa:
                float: \kappa in PME, unit in A^-1
            lmax:
                int: max L

        Output:
            Interaction tensors components
        '''
        ## pscale == 0 ? thole1 + thole2 : DEFAULT_THOLE_WIDTH`
        DEFAULT_THOLE_WIDTH = torch.tensor(5.0, dtype=torch.float32, device = dr.device)
        DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device = dr.device)
        w = torch.heaviside(pscales, torch.tensor(0.,dtype=torch.float32, device = dr.device))
        a = w * DEFAULT_THOLE_WIDTH + (1-w) * (thole1+thole2)
        
        dmp = trim_val_0(dmp)
        u = trim_val_infty(dr/dmp)

        ## au <= 50 aupi = au ;au> 50 aupi = 50
        au = a * u
        expau = torch.where(au < torch.tensor(50., dtype = torch.float32, device = dr.device), torch.exp(-au), torch.tensor(0., dtype=torch.float32, device=dr.device))

        ## compute the Thole damping factors for energies
        au2 = trim_val_infty(au*au)
        au3 = trim_val_infty(au2*au)
        au4 = trim_val_infty(au3*au)
        au5 = trim_val_infty(au4*au)
        au6 = trim_val_infty(au5*au)

        ##  Thole damping factors for energies
        thole_c   = 1.0 - expau*(1.0 + au + 0.5*au2)
        thole_d0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0)
        thole_d1  = 1.0 - expau*(1.0 + au + 0.5*au2)
        thole_q0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/18.0)
        thole_q1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0)

        rInv = 1 / dr

        rInvVec = [DIELECTRIC*(rInv**i) for i in range(0, 9)]

        alphaRVec = [(kappa*dr)**i for i in range(0, 10)]

        X = 2 * torch.exp(-alphaRVec[2]) / torch.sqrt(torch.tensor(torch.pi, dtype=torch.float32, device=dr.device))
        tmp = alphaRVec[1]
        doubleFactorial = 1
        facCount = 1
        erfAlphaR = erf(alphaRVec[1])

        # the calc_kernel using the pmap, dr is a value
        bVec = [torch.zeros_like(erfAlphaR), -erfAlphaR]

        #bVec = torch.empty(6)

        #bVec[1] = -erfAlphaR
        for i in range(2, 6):
            bVec.append(bVec[i-1] + (tmp*X/doubleFactorial))
            facCount = facCount + 2
            doubleFactorial = doubleFactorial * facCount
            tmp = tmp * 2 * alphaRVec[2]

        ## C-Uind
        cud = 2.0*rInvVec[2]*(pscales*thole_c + bVec[2])
        if lmax >= 1:
            ##  D-Uind terms
            dud_m0 = -2.0*2.0/3.0*rInvVec[3]*(3.0*(pscales*thole_d0 + bVec[3]) + alphaRVec[3]*X)
            dud_m1 = 2.0*rInvVec[3]*(pscales*thole_d1 + bVec[3] - 2.0/3.0*alphaRVec[3]*X)
        else:
            dud_m0 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)
            dud_m1 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)

        if lmax >= 2:
            udq_m0 = 2.0*rInvVec[4]*(3.0*(pscales*thole_q0 + bVec[3]) + 4/3*alphaRVec[5]*X)
            udq_m1 =  -2.0*torch.sqrt(torch.tensor(3., dtype=torch.float32, device=dr.device))*rInvVec[4]*(pscales*thole_q1 + bVec[3])
        else:
            udq_m0 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)
            udq_m1 = torch.tensor(0.0, dtype=torch.float32, device=dr.device)
        ## Uind-Uind
        udud_m0 = -2.0/3.0*rInvVec[3]*(3.0*(dscales*thole_d0 + bVec[3]) + alphaRVec[3]*X)
        udud_m1 = rInvVec[3]*(dscales*thole_d1 + bVec[3] - 2.0/3.0*alphaRVec[3]*X)
        return cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1

    if lpol:
        cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1 = calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax)

    @jit_condition()
    def calc_e_tot(qiQI, qiQJ, qiUindI, qiUindJ, lmax, lpol, cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1):
        Vij0 = cc*qiQI[0]
        Vji0 = cc*qiQJ[0]
        # C-Uind
        if lpol > 0:
            Vij0 -= cud * qiUindI[0]
            Vji0 += cud * qiUindJ[0]

        if lmax >= 1:
            # C-D
            Vij0 = Vij0 - cd*qiQI[1]
            Vji1 = -cd*qiQJ[0]
            Vij1 = cd*qiQI[0]
            Vji0 = Vji0 + cd*qiQJ[1]
            # D-D m0
            Vij1 += dd_m0 * qiQI[1]
            Vji1 += dd_m0 * qiQJ[1]
            # D-D m1
            Vij2 = dd_m1*qiQI[2]
            Vji2 = dd_m1*qiQJ[2]
            Vij3 = dd_m1*qiQI[3]
            Vji3 = dd_m1*qiQJ[3]
            # D-Uind
            if lpol > 0:
                Vij1 += dud_m0 * qiUindI[0]
                Vji1 += dud_m0 * qiUindJ[0]
                Vij2 += dud_m1 * qiUindI[1]
                Vji2 += dud_m1 * qiUindJ[1]
                Vij3 += dud_m1 * qiUindI[2]
                Vji3 += dud_m1 * qiUindJ[2]
        else:
            Vji1 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij1 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            # D-D m1
            Vij2 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji2 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij3 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji3 = torch.tensor(0., dtype=torch.float32, device=cc.device)

        if lmax >= 2:
            # C-Q
            Vij0 = Vij0 + cq*qiQI[4]
            Vji4 = cq*qiQJ[0]
            Vij4 = cq*qiQI[0]
            Vji0 = Vji0 + cq*qiQJ[4]
            # D-Q m0 
            Vij1 += dq_m0*qiQI[4]
            Vji4 += dq_m0*qiQJ[1]
            # Q-D m0 
            Vij4 -= dq_m0*qiQI[1]
            Vji1 -= dq_m0*qiQJ[4]
            # D-Q m1
            Vij2 = Vij2 + dq_m1*qiQI[5]
            Vji5 = dq_m1*qiQJ[2]
            Vij3 += dq_m1*qiQI[6]
            Vji6 = dq_m1*qiQJ[3]
            Vij5 = -(dq_m1*qiQI[2])
            Vji2 += -(dq_m1*qiQJ[5])
            Vij6 = -(dq_m1*qiQI[3])
            Vji3 += -(dq_m1*qiQJ[6])
            # Q-Q m0
            Vij4 += qq_m0*qiQI[4]
            Vji4 += qq_m0*qiQJ[4]
            # Q-Q m1
            Vij5 += qq_m1*qiQI[5]
            Vji5 += qq_m1*qiQJ[5]
            Vij6 += qq_m1*qiQI[6]
            Vji6 += qq_m1*qiQJ[6]
            # Q-Q m2
            Vij7  = qq_m2*qiQI[7]
            Vji7  = qq_m2*qiQJ[7]
            Vij8  = qq_m2*qiQI[8]
            Vji8  = qq_m2*qiQJ[8]
            # Q-Uind
            if lpol > 0:
                Vji4 += udq_m0*qiUindJ[0]
                Vij4 -= udq_m0*qiUindI[0]
                Vji5 += udq_m1*qiUindJ[1]
                Vji6 += udq_m1*qiUindJ[2]
                Vij5 -= udq_m1*qiUindI[1]
                Vij6 -= udq_m1*qiUindI[2]
        else:
            # C-Q
            Vji4 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij4 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            # D-Q m1
            Vji5 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji6 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij5 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij6 = torch.tensor(0., dtype=torch.float32, device=cc.device)
            # Q-Q m2
            Vij7  = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji7  = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij8  = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji8  = torch.tensor(0., dtype=torch.float32, device=cc.device)


        # Uind - Uind
        if lpol > 0:
            Vij1dd = udud_m0 * qiUindI[0]
            Vji1dd = udud_m0 * qiUindJ[0]
            Vij2dd = udud_m1 * qiUindI[1]
            Vji2dd = udud_m1 * qiUindJ[1]
            Vij3dd = udud_m1 * qiUindI[2]
            Vji3dd = udud_m1 * qiUindJ[2]
            Vijdd = torch.stack(( Vij1dd, Vij2dd, Vij3dd))
            Vjidd = torch.stack(( Vji1dd, Vji2dd, Vji3dd))
        else:
            Vij1dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji1dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij2dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji2dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vij3dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vji3dd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vijdd = torch.tensor(0., dtype=torch.float32, device=cc.device)
            Vjidd = torch.tensor(0., dtype=torch.float32, device=cc.device)


        if lmax == 0:
            Vij = Vij0
            Vji = Vji0
        elif lmax == 1:
            Vij = torch.stack((Vij0, Vij1, Vij2, Vij3))
            Vji = torch.stack((Vji0, Vji1, Vji2, Vji3))
        elif lmax == 2:
            Vij = torch.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
            Vji = torch.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))
        else:
            raise ValueError(f"Invalid lmax {lmax}. Valid values are 0, 1, 2")
        if lpol > 0:
            return 0.5 * (torch.sum(qiQJ*Vij) + torch.sum(qiQI*Vji)) + 0.5 * (torch.sum(qiUindJ*Vijdd) + torch.sum(qiUindI*Vjidd))
        else:
            return 0.5 * (torch.sum(qiQJ*Vij) + torch.sum(qiQI*Vji))

    if lpol == True:
        e_tot = calc_e_tot(qiQI, qiQJ, qiUindI, qiUindJ, lmax, torch.tensor(1.,dtype=torch.float32, device=cc.device), cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1)
    else:
        e_tot = calc_e_tot(qiQI, qiQJ, qiUindI, qiUindJ, lmax, torch.tensor(0.,dtype=torch.float32, device=cc.device), cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, None, None, None, None, None, None, None)
    return e_tot

def setup_ewald_parameters(
    rc: float,
    ethresh: torch.tensor,
    box: Optional[torch.tensor] = None,
    spacing: Optional[float] = None,
    method: str = 'openmm'
) -> Tuple[float, int, int, int]:
    '''
    Given the cutoff distance, and the required precision, determine the parameters used in
    Ewald sum, including: kappa, K1, K2, and K3.
    

    Parameters:
    ----------
    rc: float
        The cutoff distance, in nm
    ethresh: float
        Required energy precision, in kJ/mol
    box: ndarray, optional
        3*3 matrix, box size, a, b, c arranged in rows, used in openmm method
    spacing: float, optional
        fourier spacing to determine K, used in gromacs method
    method: str
        Method to determine ewald parameters. Valid values: "openmm" or "gromacs".
        If openmm, the algorithm can refer to http://docs.openmm.org/latest/userguide/theory.html
        If gromacs, the algorithm is adapted from gromacs source code

    Returns
    -------
    kappa, K1, K2, K3: (float, int, int, int)
        float, the attenuation factor
    K1, K2, K3:
        integers, sizes of the k-points mesh
    '''
    if method == "openmm":
        kappa = torch.sqrt(-torch.log(2 * ethresh)) / rc
        K1 = torch.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
        K2 = torch.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
        K3 = torch.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)
        return kappa, int(K1), int(K2), int(K3)
    elif method == "gromacs":
        # determine kappa
        kappa = 5.0
        i = 0
        while erfc(kappa * rc) > ethresh:
            i += 1
            kappa *= 2

        n = i + 60
        low = 0.0
        high = kappa
        for k in range(n):
            kappa = (low + high) / 2
            if erfc(kappa * rc) > ethresh:
                low = kappa
            else:
                high = kappa
        # determine K
        K1 = int(torch.ceil(box[0, 0] / spacing))
        K2 = int(torch.ceil(box[1, 1] / spacing))
        K3 = int(torch.ceil(box[2, 2] / spacing))
        return kappa, K1, K2, K3
    else:
        raise ValueError(
            f"Invalid method: {method}."
            "Valid methods: 'openmm', 'gromacs'"
        )


def energy_pme(positions, box, pairs, 
        Q_local, Uind_global, pol, tholes, 
        mScales, pScales, dScales,
        construct_local_frame_fn, kappa, K1, K2, K3, lmax, lpol, n_mesh, shifts, B, ldmp, lpme=True):
    '''
    This is the top-level wrapper for multipole PME 

    Input:
        position: 
            Na * 3: positions
        box:
            3 * 3: box
        Q_local:
            Na * (lmax+1)^2: harmonic multipoles of each site in local frame
        Uind_global:
            Na * 3: the induced dipole moment, in GLOBA: CARTESIAN!
        pol: 
            (Na,) float: the polarizability of each site, unit in A**3
        tholes:
            (Na,) float: the thole damping widths for each atom, it's dimensionless, default is 8 according to MPID paper
        mScales, pScales, dScales:
            (Nexcl,): multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
            for permanent-parmenent, perment-induced, induced-induced interactions
        pairs:
            Np * 3: interacting pair indices and topology distance
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        construct_local_frame_in:
            function: local frame constructors, from generate_local_frame_constructor
        pme_recip:
            function: see recip.py, a reciprocal space calculator
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculatioins
        lmax:
            int: maximum L
        lpol:
            bool: if polarizable or not? if yes, 1, otherwise 0 
        lpme:
            bool: doing pme? If false, then turn off reciprocal space and set kappa = 0
    
    Output:
        energy: total pme energy
    '''
    # in this simplest demo, we only do the coulomb interation at no pbc condition (do not use the PME)
    if lmax > 0:
        if construct_local_frame_fn == None:
            Q_global = Q_local
        else:
            local_frames = construct_local_frame_fn(positions, box)
            Q_global = rot_local2global(Q_local, local_frames, lmax)
    else:
        if lpol:
            # if fixed multipole only contains charge, and it's polarizable, then expand Q matrix
            dips = torch.zeros((Q_local.shape[0], 3), dtype=torch.float32, device=positions.device)
            Q_global = torch.hstack((Q_local, dips))
            lmax = 1
        else:
            Q_global = Q_local

    # note we assume when lpol is True, lmax should be >= 1
    if lpol:
        # convert Uind to global harmonics, in accord with Q_global
        U_ind = torch.matmul(C1_c2h,Uind_global.T).T 
        Q_global_tot = Q_global.clone()
        Q_global_tot[:, 1:4] = Q_global_tot[:,1:4] + U_ind
    else:
        Q_global_tot = Q_global
    if lpme is False:
        kappa = torch.tensor(0., dtype=torch.float32, device=positions.device)

    # first, we need to consider the lpol in pme_real 
    if lpol:
        ene_real = pme_real(positions, box, pairs, Q_global, U_ind, pol, tholes,
                          mScales, pScales, dScales, kappa, lmax, True, B, ldmp)
    else:
        ene_real = pme_real(positions, box, pairs, Q_global, None, None, None, 
                          mScales, None, None, kappa, lmax, False, B, ldmp)
    def pme_self(Q_h, kappa, lmax):
        '''
        This function calculates the PME self energy
    
        Inputs:
            Q:
                Na * (lmax+1)^2: harmonic multipoles, local or global does not matter
            kappa:
                float: kappa used in PME

        Output:
            ene_self:
                float: the self energy
        '''
        DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device=Q_h.device)
        n_harms = (lmax + 1) ** 2 
        l_list = torch.tensor([0,1,1,1,2,2,2,2,2], dtype=torch.float32, device=Q_h.device)[:n_harms]
        l_fac2 = torch.tensor([1,3,3,3,15,15,15,15,15], dtype=torch.float32, device=Q_h.device)[:n_harms]
        factor = kappa/torch.sqrt(torch.tensor(torch.pi)) * (2*kappa**2)**l_list / l_fac2
        return - torch.sum(factor[np.newaxis] * Q_h**2) * DIELECTRIC

    @jit_condition()
    def pol_penalty(U_ind, pol):
        '''
        The energy penalty for polarization of each site, currently only supports isotropic polarization:
    
        Inputs:
            U_ind:
                Na * 3 float: induced dipoles, in isotropic polarization case, cartesian or harmonic does not matter
            pol:
                (Na,) float: polarizability
        '''
        
        # this is to remove the singularity when pol=0
        DIELECTRIC = torch.tensor(1389.35455846, dtype=torch.float32, device=U_ind.device)
        #@jit_condition()
        #def trim_val_0(x):
        #    return torch.where(x < 1e-8, torch.tensor(1e-8), x)
        pol_pi = torch.where(pol < torch.tensor(1e-8, dtype=torch.float32, device=U_ind.device), torch.tensor(1e-8, dtype=torch.float32, device=U_ind.device), pol)
        #pol_pi = trim_val_0(pol)
        # pol_pi = pol/(jnp.exp((-pol+1e-08)*1e10)+1) + 1e-08/(jnp.exp((pol-1e-08)*1e10)+1)
        return torch.sum(0.5/pol_pi*(U_ind**2).T) * DIELECTRIC

    if lpme:
        ene_recip = pme_recip(torch.tensor(1, dtype=torch.int32, device=U_ind.device), kappa, False, K1, K2, K3, positions, box, Q_global_tot, n_mesh, shifts, lmax)
        ene_self = pme_self(Q_global_tot, kappa, lmax)
        if lpol:
            ene_self += pol_penalty(U_ind, pol)
        return ene_real + ene_recip + ene_self

    else:
        if lpol:
            ene_self = pol_penalty(U_ind, pol)
        else:
            ene_self = 0.0
        return ene_real + ene_self 

#first we define the cov_map from the topo infor
def get_axis_idx(ii,conn_atom,ele):
    ele = [atomic_num[u] for u in ele]
    z_idx = None; x_idx = None
    nei_0 = conn_atom[ii]
    if len(nei_0) == 1:
        z_idx = nei_0[0]
        nei_1 = conn_atom[z_idx]
        nei_ele = [ele[u] for u in nei_1]
        nei_1 = np.array(nei_1)[np.argsort(nei_ele)]
        for uu in nei_1:
            if uu != ii and x_idx == None:
                x_idx = uu
    else:
        nei_ele = [ele[u] for u in nei_0]
        z_idx = nei_0[np.argsort(nei_ele)[-1]]
        x_idx = nei_0[np.argsort(nei_ele)[-2]]
    assert(z_idx != None and x_idx !=None)
    return z_idx, x_idx


if __name__ == '__main__':
    data = np.load('100K_properties.npz',allow_pickle=True)
    num_idx = 26
    positions = data['coord'][num_idx]; topo = data['topo'][num_idx]; Q_q = np.array(data['charge'][num_idx]).reshape(-1,1)
    Q_dipole = data['dipole'][num_idx]; Q_qua = data['quadrupole'][num_idx]; Polar = data['polar'][num_idx]
    c6 = data['c6'][num_idx]; c8 = data['c8'][num_idx]; c10 = data['c10'][num_idx]
    symbol = data['symbol'][num_idx]
    
    conn_atom = {}
    for pair in topo:
        conn_atom[pair[0]] = []
    for pair in topo:
        conn_atom[pair[0]] = []
    for pair in topo:
        conn_atom[pair[0]].append(pair[1])
    
    import ase
    atomic_num = ase.data.atomic_numbers
    atoms = {'positions':positions,'bonds':topo,'Q_q':Q_q}
    cov_map = build_covalent_map(atoms, 6)
    pair_full = []
    for na in range(len(atoms['positions'])):
        for nb in range(na + 1, len(atoms['positions'])):
            pair_full.append([na, nb, 0])
    pair_full = np.array(pair_full, dtype=int)
    # temporarily, only need full pair 
    pair_full[:, 2] = cov_map[pair_full[:,0], pair_full[:,1]]
    
    pairs = torch.tensor(pair_full,requires_grad=False)
    mscales = torch.tensor([0., 0., 0., 0., 1., 1.], dtype=torch.float32, requires_grad=False)
    box = torch.tensor([[50.,0.,0.],[0.,50.,0.],[0.,0.,50.]], dtype=torch.float32, requires_grad=False)
    positions = torch.tensor(positions, dtype=torch.float32, requires_grad=True)
    rc = torch.tensor(5.0,dtype=torch.float32, requires_grad=False)
    ethresh = torch.tensor(5e-4,dtype=torch.float32,requires_grad=False)
    # here we need to calculate the local 
    Q_q = torch.tensor(Q_q, dtype=torch.float32, requires_grad=False)
    Q_dipole = torch.tensor(Q_dipole, dtype=torch.float32, requires_grad=False)
    Q_qua = torch.tensor(Q_qua, dtype=torch.float32, requires_grad=False)
    
    # temp, for code verify, we didn't do local 
    Q_local = torch.hstack((Q_q,Q_dipole,Q_qua))
    # get the axis_type and axis_indices; axis_indices should be pytorch tensor
    axis_types = []; axis_indices = []; ZThenX = 0; yaxis=-1
    for ii in range(len(symbol)):
        axis_types.append(ZThenX)
        zaxis, xaxis = get_axis_idx(ii, conn_atom, symbol)
        axis_indices.append([zaxis,xaxis,yaxis])
    axis_types = np.array(axis_types); axis_indices = np.array(axis_indices)
    axis_types = torch.tensor(axis_types, dtype=torch.int, requires_grad=False)
    axis_indices = torch.tensor(axis_indices, dtype=torch.int,requires_grad=False)
    
    construct_local_frame_fn  = generate_construct_local_frames(axis_types, axis_indices)
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    
    pme_order = 6; lmax = 2
    ######################################################################
    pme_recip = generate_pme_recip(Ck_1, kappa, False, pme_order, K1, K2, K3, lmax)
    e = energy_pme(positions, box, pairs,
        Q_local, None, None, None, 
        mscales, None, None, 
        construct_local_frame_fn, pme_recip, kappa, K1, K2, K3, 2, False, lpme=True)
    grad = torch.autograd.grad(outputs=e,inputs=positions)

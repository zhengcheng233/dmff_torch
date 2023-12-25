import numpy as np
import torch
from dmff_torch.constants import DIELECTRIC, SQRT_PI as sqrt_pi
from torch import erf, erfc
from dmff_torch.utils import jit_condition
from functorch import vmap
from functools import partial

def pme_recip(Ck_fn, kappa, gamma, K1, K2, K3, positions, box, Q, n_mesh, shifts, lmax):
    '''
    The generated pme_recip space calculator
    kappa, pme_order, K1, K2, K3, and lmax are passed and fixed when the calculator is generated
    '''
        
    @jit_condition()
    def Ck_1(ksq, kappa, V):
        return 2*torch.pi/V/ksq * torch.exp(-ksq/4/kappa**2)

    @jit_condition()
    def Ck_6(ksq, kappa, V):
        sqrt_pi = torch.sqrt(torch.tensor(3.1415926535, dtype=ksq.dtype,device=ksq.device))
        x2 = ksq / 4 / kappa**2
        x = torch.sqrt(x2)
        x3 = x2 * x
        exp_x2 = torch.exp(-x2)
        f = (1 - 2*x2)*exp_x2 + 2*x3*sqrt_pi*erfc(x)
        return sqrt_pi*torch.pi/2/V*kappa**3 * f / 3

    @jit_condition()
    def Ck_8(ksq, kappa, V):
        sqrt_pi = torch.sqrt(torch.tensor(3.1415926535, dtype=ksq.dtype,device=ksq.device))
        x2 = ksq / 4 / kappa**2
        x = torch.sqrt(x2)
        x4 = x2 * x2
        x5 = x4 * x
        exp_x2 = torch.exp(-x2)
        f = (3 - 2*x2 + 4*x4)*exp_x2 - 4*x5*sqrt_pi*erfc(x)
        return sqrt_pi*torch.pi/2/V*kappa**5 * f / 45
    
    @jit_condition()
    def Ck_10(ksq, kappa, V):
        sqrt_pi = torch.sqrt(torch.tensor(3.1415926535, dtype=ksq.dtype,device=ksq.device))
        x2 = ksq / 4 / kappa**2
        x = torch.sqrt(x2)
        x4 = x2 * x2
        x6 = x4 * x2
        x7 = x6 * x
        exp_x2 = torch.exp(-x2)
        f = (15 - 6*x2 + 4*x4 - 8*x6)*exp_x2 + 8*x7*sqrt_pi*erfc(x)
        return sqrt_pi*torch.pi/2/V*kappa**7 * f / 1260

    @jit_condition()
    def get_recip_vectors(N, box):
        """
        Computes reciprocal lattice vectors of the grid
        
        Input:
            N:
                (3,)-shaped array, (K1, K2, K3)
            box:
                3 x 3 matrix, box parallelepiped vectors arranged in TODO rows or columns?
                
        Output: 
            Nj_Aji_star:
                3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                (lattice vectors arranged in rows)
        """
        Nj_Aji_star = (N.reshape((1,3)) * torch.linalg.inv(box)).transpose(0,1)
        return Nj_Aji_star 

    @jit_condition()
    def u_reference(R_a, Nj_Aji_star):
        '''
        Each atom is meshed to dispersion_ORDER**3 points on the m-meshgrid.
        This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates,
        and the corresponding values of xyz fractional displacements from real coordinate to the reference point.

        Inputs:
            R_a:
                N_a * 3 matrix containing positions of sites
            Nj_Aji_star:
                3 x 3 matrix, the first index denotes reciprocal lattice vector, the second index is the component xyz.
                (lattice vectors arranged in rows)

        Outputs:
            m_u0:
                N_a * 3 matrix, positions of the reference points of R_a on the m-meshgrid
            u0:
                N_a * 3 matrix, (R_a - R_m)*a_star values
        '''
        R_in_m_basis = torch.einsum("ij,kj->ki", Nj_Aji_star, R_a)
        m_u0 = torch.ceil(R_in_m_basis)#.to(torch.int)
        u0 = (m_u0 - R_in_m_basis) + torch.tensor(3.,dtype=torch.float32,device=m_u0.device)
        return m_u0, u0

    # we can add vmap later
    @jit_condition()
    def bspline(u):
        '''
        Computes the cardinal B-spline function 
        '''
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        u5 = u ** 5
        u_less_1 = u - 1
        u_less_1_p5 = u_less_1 ** 5
        u_less_2 = u - 2
        u_less_2_p5 = u_less_2 ** 5
        u_less_3 = u - 3
        u_less_3_p5 = u_less_3 ** 5
        conditions = [
            torch.logical_and(u >= 0., u < 1.),
            torch.logical_and(u >= 1., u < 2.),
            torch.logical_and(u >= 2., u < 3.),
            torch.logical_and(u >= 3., u < 4.),
            torch.logical_and(u >= 4., u < 5.),
            torch.logical_and(u >= 5., u < 6.)
            ]
        outputs = [
            u5 / 120,
            u5 / 120 - u_less_1_p5 / 20,
            u5 / 120 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
            u5 / 120 - u_less_3_p5 / 6 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
            u5 / 24 - u4 + 19 * u3 / 2 - 89 * u2 / 2 + 409 * u / 4 - 1829 / 20,
            -u5 / 120 + u4 / 4 - 3 * u3 + 18 * u2 - 54 * u + 324 / 5
            ]
        return torch.sum(torch.stack([condition * output for condition, output in zip(conditions, outputs)]),
                     dim=0)
    
    @jit_condition()
    def bspline_prime(u):
        '''
        Computes first derivative of the cardinal B-spline function
        '''
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4

        u_less_1 = u - 1
        u_less_1_p4 = u_less_1 ** 4

        u_less_2 = u - 2
        u_less_2_p4 = u_less_2 ** 4

        conditions = [
               torch.logical_and(u >= 0., u < 1.),
               torch.logical_and(u >= 1., u < 2.),
               torch.logical_and(u >= 2., u < 3.),
               torch.logical_and(u >= 3., u < 4.),
               torch.logical_and(u >= 4., u < 5.),
               torch.logical_and(u >= 5., u < 6.)
               ]

        outputs = [
                u4 / 24,
                u4 / 24 - u_less_1_p4 / 4,
                u4 / 24 + 5 * u_less_2_p4 / 8 - u_less_1_p4 / 4,
                -5 * u4 / 12 + 6 * u3 - 63 * u2 / 2 + 71 * u - 231 / 4,
                5 * u4 / 24 - 4 * u3 + 57 * u2 / 2 - 89 * u + 409 / 4,
                -u4 / 24 + u3 - 9 * u2 + 36 * u - 54
                ]

        return torch.sum(torch.stack([condition * output for condition, output in zip(conditions, outputs)]),
                     dim=0)

    @jit_condition()
    def bspline_prime2(u):
        '''
        Computes second derivate of the cardinal B-spline function
        ''' 
        u2 = u ** 2
        u3 = u ** 3
        u_less_1 = u - 1

        conditions = [
                torch.logical_and(u >= 0., u < 1.),
                torch.logical_and(u >= 1., u < 2.),
                torch.logical_and(u >= 2., u < 3.),
                torch.logical_and(u >= 3., u < 4.),
                torch.logical_and(u >= 4., u < 5.),
                torch.logical_and(u >= 5., u < 6.)
                ]

        outputs = [
                u3 / 6,
                u3 / 6 - u_less_1 ** 3,
                5 * u3 / 3 - 12 * u2 + 27 * u - 19,
               -5 * u3 / 3 + 18 * u2 - 63 * u + 71,
                5 * u3 / 6 - 12 * u2 + 57 * u - 89,
               -u3 / 6 + 3 * u2 - 18 * u + 36
                ]

        return torch.sum(torch.stack([condition * output for condition, output in zip(conditions, outputs)]),
                     dim=0)

    @jit_condition()
    def theta_eval(M_u):
        '''
        Evaluates the value of theta given 3D u values at ... points

        Input:
            u:
                ... x 3 matrix

        Output:
            theta:
                ... matrix
        '''
        theta = torch.prod(M_u, dim = 1)
        return theta 

    @jit_condition()
    def thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u):
        '''
        First derivative of theta with respect to x,y,z directions

        Input:
            u
            Nj_Aji_star:
                reciprocal lattice vectors

        Output:
            N_a * 3 matrix
        '''

        div = torch.stack((
            Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
            Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
            Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            )).transpose(0,1)
        return torch.einsum("ij,kj->ki", -Nj_Aji_star, div)

    @jit_condition()
    def theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
        """
        compute the 3 x 3 second derivatives of theta with respect to xyz
        
        Input:
            u
            Nj_Aji_star
        
        Output:
            N_A * 3 * 3
        """
        div_00 = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
        div_11 = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
        div_22 = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]

        div_01 = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
        div_02 = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
        div_12 = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]
        
        div_10 = div_01
        div_20 = div_02
        div_21 = div_12

        div = torch.transpose(torch.stack((
            torch.stack((div_00, div_01, div_02)),
            torch.stack((div_10, div_11, div_12)),
            torch.stack((div_20, div_21, div_22)),
        )),0,2)
        return torch.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)

    @jit_condition()
    def sph_harmonics_GO(u0, Nj_Aji_star, n_mesh, shifts, lmax):
        '''
        Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
        00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
        Currently supports lmax <= 2
    
        Inputs:
            u0: 
                a N_a * 3 matrix containing all positions
            Nj_Aji_star:
                reciprocal lattice vectors in the m-grid
            lmax:
                int: max L
    
        Output: 
            harmonics: 
                a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
                evaluated at 6*6*6 integer points about reference points m_u0 
        '''
        n_harm = int((lmax + 1)**2)

        N_a = u0.shape[0]
        # mesh points around each site
        #u = (u0[:, None, :] + shifts).reshape((N_a*n_mesh, 3))
        u = torch.reshape(torch.unsqueeze(u0, 1) + shifts, (int(N_a*n_mesh), 3))
        # bspline may have little different value
        M_u = bspline(u)
        theta = theta_eval(M_u)
        if lmax == 0:
            return theta.reshape(N_a, n_mesh, n_harm)   
        # dipole
        Mprime_u = bspline_prime(u)
        thetaprime = thetaprime_eval(u, Nj_Aji_star, M_u, Mprime_u)
        harmonics_1 = torch.stack(
            [theta,
            thetaprime[:, 2],
            thetaprime[:, 0],
            thetaprime[:, 1]],
            dim = -1
        )

        if lmax == 1:
            return harmonics_1.reshape(N_a, n_mesh, n_harm)

        # quadrapole
        M2prime_u = bspline_prime2(u)
        theta2prime = theta2prime_eval(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u)
        rt3 = torch.sqrt(torch.tensor(3.,dtype=torch.float32, device=u0.device))
        harmonics_2 = torch.hstack(
            [harmonics_1,
            torch.stack([(3*theta2prime[:,2,2] - torch.diagonal(theta2prime,dim1=1,dim2=2).sum(dim=1)) / 2,
            rt3 * theta2prime[:, 0, 2],
            rt3 * theta2prime[:, 1, 2],
            rt3/2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1]),
            rt3 * theta2prime[:, 0, 1]], dim = 1)]
        )
        if lmax == 2:
            return harmonics_2.reshape(N_a, n_mesh, n_harm)
        else:
            raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

    @jit_condition()
    def Q_m_peratom(Q, sph_harms, n_mesh, lmax):
        """
        Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983
        
        Inputs:
            Q: 
                N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
            sph_harms:
                N_a, 216, (l+1)**2
            lmax:
                int: maximal L
        
        Output:
            Q_m_pera:
               N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
        """
        N_a = sph_harms.shape[0]

        if lmax > 2:
            raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

        Q_dbf = Q[:, 0:1]
        if lmax >= 1:
            Q_dbf = torch.hstack([Q_dbf, Q[:,1:4]])
        if lmax >= 2:
            Q_dbf = torch.hstack([Q_dbf, Q[:,4:9]/3])

        Q_m_pera = torch.sum(Q_dbf[:,None,:]* sph_harms, dim=2)
        assert Q_m_pera.shape == (int(N_a), int(n_mesh))
        return Q_m_pera

    @jit_condition()
    def Q_mesh_on_m(Q_mesh_pera, m_u0, N, shifts):
        """
        Reduce the local Q_m_peratom into the global mesh
        
        Input:
            Q_mesh_pera, m_u0, N
            
        Output:
            Q_mesh: 
                Nx * Ny * Nz matrix
        """
        # torch.fmod is different from np.mod in negative num
        indices_arr = torch.fmod(m_u0[:,None,:]+shifts+N[None,None,:]*10, N[None, None, :])
        ### in jax version, the trick implementation without using for loop
        ### in torch versioin, this trick is not supported, we can use index_add_ or pytorch-scatter,
        ### both of them only work on 1 dim. 
        Q_mesh = torch.zeros(N[0]*N[1]*N[2], dtype=torch.float32, device=m_u0.device)
        indices_0 = indices_arr[:,:,0].flatten()
        indices_1 = indices_arr[:,:,1].flatten()
        indices_2 = indices_arr[:,:,2].flatten()
        flat_indices = (indices_0 * N[1] * N[2] + indices_1 * N[2] + indices_2).to(torch.int32)
        Q_mesh.index_add_(0, flat_indices, Q_mesh_pera.view(-1))
        Q_mesh = Q_mesh.view((int(N[0]),int(N[1]),int(N[2])))
        return Q_mesh 

    def setup_kpts_integer(N,N_list):
        """
        Outputs:
            kpts_int:
                n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
        """
        #############################################################
        # I am not sure wheter tolist will cause code unefficiency
        #############################################################
        #N_half = N.reshape(3).tolist()

        kx, ky, kz = [torch.roll(torch.arange(- (N_list[i] - 1) // 2, (N_list[i] + 1) // 2, dtype=torch.int32, device=N.device), - (N_list[i] - 1) // 2) for i in range(3)]
        kpts_int = torch.hstack([ki.flatten()[:,None] for ki in torch.meshgrid(kz, kx, ky, indexing='ij')])
        return kpts_int

    @jit_condition()
    def setup_kpts(box, kpts_int):
        '''
        This function sets up the k-points used for reciprocal space calculations
        
        Input:
            box:
                3 * 3, three axis arranged in rows
            kpts_int:
                n_k * 3 matrix

        Output:
            kpts:
                4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
        '''
        # in this array, a*, b*, c* (without 2*pi) are arranged in column
        box_inv = torch.linalg.inv(box)
        # K * 3, coordinate in reciprocal space
        kpts = 2 * torch.pi * torch.matmul(kpts_int.float(), box_inv)
        ksq = torch.sum(kpts**2, dim=1)
        # 4 * K
        kpts = torch.hstack((kpts, ksq[:, None])).transpose(0,1)
        #######################################################################
        # may be more faster
        #kpts = torch.concatenate((kpts.transpose(0,1), ksq[None:]),axis=0)
        #######################################################################
        return kpts

    def spread_Q(positions, box, Q, n_mesh, shifts, lmax, N):
        '''
        This is the high level wrapper function, in charge of spreading the charges/multipoles on grid

        Input:
            positions:
                Na * 3: positions of each site
            box: 
                3 * 3: box
            Q:
                Na * (lmax+1)**2: the multipole of each site in global frame

        Output:
            Q_mesh:
                K1 * K2 * K3: the meshed multipoles
            
        '''
        Nj_Aji_star = get_recip_vectors(N, box)
        # For each atom, find the reference mesh point, and u position of the site
        m_u0, u0 = u_reference(positions, Nj_Aji_star)
        # find out the STGO values of each grid point
        sph_harms = sph_harmonics_GO(u0, Nj_Aji_star, n_mesh, shifts, lmax)
        # find out the local meshed values for each site
        Q_mesh_pera = Q_m_peratom(Q, sph_harms, n_mesh, lmax)
        return Q_mesh_on_m(Q_mesh_pera, m_u0, N, shifts)

    # spread Q
    N = torch.tensor([K1, K2, K3], dtype=torch.int32, device=positions.device)
    Q_mesh = spread_Q(positions, box, Q, n_mesh, shifts, lmax, N)
    N = N.reshape(1, 1, 3)
    kpts_int = setup_kpts_integer(N,[K1,K2,K3])
    kpts = setup_kpts(box, kpts_int)

    pme_order = torch.tensor(6, dtype=torch.int32, device=positions.device)
    m = torch.linspace(-pme_order//2+1, pme_order//2-1, pme_order-1, dtype=torch.float32, device=positions.device).reshape(pme_order-1, 1, 1)
    
    @jit_condition()
    def calc_s_k(m, kpts_int, box, Q_mesh, N):
        theta_k = torch.prod(
            torch.sum(
                bspline(m + 3) * torch.cos(2*torch.pi*m*kpts_int[None] / N),
                dim = 0
                ),
            dim = 1
            )
        V = torch.linalg.det(box)
    
        S_k = torch.fft.fftn(Q_mesh).flatten()
        return theta_k, S_k, V

    theta_k, S_k, V = calc_s_k(m, kpts_int, box, Q_mesh, N)
    
    # for electrostatic, need to exclude gamma point
    # for dispersion, need to include gamma point
    if not gamma:
        if Ck_fn < 2:
            C_k = Ck_1(kpts[3, 1:], kappa, V)
        elif Ck_fn < 7:
            C_k = Ck_6(kpts[3, 1:], kappa, V)
        elif Ck_fn < 9:
            C_k = Ck_8(kpts[3, 1:], kappa, V)
        elif Ck_fn < 11:
            C_k = Ck_10(kpts[3, 1:], kappa, V)
        E_k = C_k *  torch.abs(S_k[1:] / theta_k[1:])**2
    else:
        if Ck_fn < 2:
            C_k = Ck_1(kpts[3, :], kappa, V)
        elif Ck_fn < 7:
            C_k = Ck_6(kpts[3, :], kappa, V)
        elif Ck_fn < 9:
            C_k = Ck_8(kpts[3, :], kappa, V)
        elif Ck_fn < 11:
            C_k = Ck_10(kpts[3, :], kappa, V)
        E_k = C_k * torch.abs(S_k / theta_k)**2

    if not gamma:
        return torch.sum(E_k) * DIELECTRIC
    else:
        return torch.sum(E_k)


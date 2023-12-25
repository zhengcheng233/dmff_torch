from functools import partial
from functorch import vmap 
from dmff_torch.spatial import v_pbc_shift
from dmff_torch.utils import jit_condition
import torch

DIELECTRIC = 1389.35455846

# the torch give the Error: vmap: It looks like you're calling .item() on a Tensor. We don't support vmap over calling .item() on a Tensor

@partial(vmap, in_dims=(0, None), out_dims = (0))
def distribute_v3(pos, index):
    return torch.index_select(pos, 0, index)

def distribute_scalar(params, index):
    return torch.index_select(params, 0, index)

@partial(vmap, in_dims=(0, None), out_dims = (0))
def distribute_multipoles(multipoles, index):
    return torch.index_select(multipoles, 0, index)

@partial(vmap, in_dims=(0, None), out_dims = (0))
def distribute_dispcoeff(c_list, index):
    return torch.index_select(c_list, 0, index)

#def distribute_v3(pos, index):
#    return torch.gather(pos, 0, index.view(-1,1).expand(-1, pos.shape[1]))
#def distribute_scalar(params, index):
#    return torch.gather(params, 0, index)
#def distribute_multipoles(multipoles, index):
#    return torch.gather(multipoles, 0, index.view(-1,1).expand(-1, multipoles.shape[1]))
#def distribute_dispcoeff(c_list, index):
#    return torch.gather(c_list, 0, index.view(-1,1).expand(-1, c_list.shape[1]))

def generate_pairwise_interaction(pair_int_kernel, static_args):
    ''' 
    This is a calculator generator for pairwise interaction

    Input: 
        pair_int_kernel:
            function type (dr, m, p1i, p1j, p2i, p2j) -> energy : the vectorized kernel function, 
            dr is the distance, m is the topological scaling factor, p1i, p1j, p2i, p2j are pairwise parameters

        static_args:
            dict: a dictionary that stores all static global parameters (such as lmax, kappa, etc)

    Output:
        pair_int:
            function type (positions, box, pairs, mScales, p1, p2, ...) -> energy
            The pair interaction calculator. p1, p2 ... involved atomic parameters, the order should be consistent
            with the order in kernel
    '''

    def pair_int(positions, box, pairs, mScales, *atomic_params):
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
        
        ri = distribute_v3(positions.T, pairs[:, 0]).T
        rj = distribute_v3(positions.T, pairs[:, 1]).T

        nbonds = pairs[:, 2]
        indices = (nbonds + (mScales.shape[0] - 1)) % mScales.shape[0]
        mscales = distribute_scalar(mScales, indices)

        buffer_scales = pair_buffer_scales(pairs)
        mscales = mscales * buffer_scales

        box_inv = torch.linalg.inv(box)
        dr = ri - rj
        dr = v_pbc_shift(dr, box, box_inv)
        dr = torch.linalg.norm(dr, axis=1)

        pair_params = []
        for i, param in enumerate(atomic_params):
            pair_params.append(distribute_scalar(param, pairs[:, 0]))
            pair_params.append(distribute_scalar(param, pairs[:, 1]))

        energy = torch.sum(pair_int_kernel(dr, mscales, *pair_params) * buffer_scales)
        return energy 
    return pair_int


# different kinds of pair_int_kernel
@vmap
def TT_damping_qq_c6_kernel(dr, m, ai, aj, bi, bj, qi, qj, ci, cj):
    # include the a*exp(-br); q-q and c6-c6
    a = torch.sqrt(ai * aj)
    b = torch.sqrt(bi * bj)
    c = ci * cj
    q = qi * qj
    r = dr * 1.889726878 # convert to bohr
    br = b * r
    br2 = br * br
    br3 = br2 * br
    br4 = br3 * br
    br5 = br4 * br
    br6 = br5 * br
    exp_br = torch.exp(-br)
    f = 2625.5 * a * exp_br \
        + (-2625.5) * exp_br * (1+br) * q / r \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720) * c / dr**6
    return f * m

@vmap
@jit_condition()
def TT_damping_qq_disp_kernel(dr, m, ex_ai, ex_aj, es_ai, es_aj, pol_ai, pol_aj, disp_ai, disp_aj, dhf_ai, dhf_aj, bi, bj, qi, qj, c6i, c6j, c8i, c8j, c10i, c10j):
    ############################################################
    # include a*exp(-br), q-q, c6-c6, c8-c8, c10-c10 interaction
    # in future, if we predict a, b, c via nn, all of them are 
    # sqrt(a), sqrt(b), sqrt(c), now is sqrt(a), b and sqrt(c)
    ############################################################
    a_ex = (ex_ai * ex_aj)
    a_es = (es_ai * es_aj)
    a_pol = (pol_ai * pol_aj)
    a_disp = (disp_ai * disp_aj)
    a_dhf = (dhf_ai * dhf_aj)
    b = torch.sqrt(bi * bj); q = qi * qj 
    c6 = c6i * c6j; c8 = c8i * c8j; c10 = c10i * c10j 
    br = b * dr 
    br2 = br * br 
    br3 = br2 * br
    br4 = br3 * br
    br5 = br4 * br
    br6 = br5 * br
    br7 = br6 * br
    br8 = br7 * br
    br9 = br8 * br
    br10 = br9 * br 
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
    ###################################################################
    # attention to the unit of sr, damp disp and damp q-q interaction
    # unit of slater-isa is kj/mol; unit of disp is hartree as the 
    # c6 is hartree/A**6 and dr is A; q-q interaction is hartree if r 
    # is bohr, here r is A, so DIELECTRIC is needed to convert kj/mol 
    ####################################################################
    #f = (a_ex + a_es + a_pol + a_disp + a_dhf) * P * exp_br / 2625.5 # 9.3556 
    #f = (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) # 200783 
    #f = - exp_br * (1+br) * q / dr / DIELECTRIC # 0.0311 
    f = (a_ex + a_es + a_pol + a_disp + a_dhf) * P * exp_br / 2625.5 + (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) - exp_br * (1+br) * q / dr * 0.529177
    return f * m 

@vmap
@jit_condition()
def TT_damping_qq_kernel(dr, m, ex_ai, ex_aj, es_ai, es_aj, pol_ai, pol_aj, disp_ai, disp_aj, dhf_ai, dhf_aj, bi, bj, qi, qj):
    ############################################################
    # include a*exp(-br), q-q, c6-c6, c8-c8, c10-c10 interaction
    # in future, if we predict a, b, c via nn, all of them are 
    # sqrt(a), sqrt(b), sqrt(c), now is sqrt(a), b and sqrt(c)
    ############################################################
    a_ex = (ex_ai * ex_aj)
    a_es = (es_ai * es_aj)
    a_pol = (pol_ai * pol_aj)
    a_disp = (disp_ai * disp_aj)
    a_dhf = (dhf_ai * dhf_aj)
    b = torch.sqrt(bi * bj); q = qi * qj 
    br = b * dr     
    br2 = br * br
    exp_br = torch.exp(-br)
    P = 1/3 * br2 + br + 1
    ###################################################################
    # attention to the unit of sr, damp disp and damp q-q interaction
    # unit of slater-isa is kj/mol; unit of disp is hartree as the 
    # c6 is hartree/A**6 and dr is A; q-q interaction is hartree if r 
    # is bohr, here r is A, so DIELECTRIC is needed to convert kj/mol 
    ####################################################################
    #f = (a_ex + a_es + a_pol + a_disp + a_dhf) * P * exp_br / 2625.5 # 9.3556 
    #f = (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) # 200783 
    #f = - exp_br * (1+br) * q / dr / DIELECTRIC # 0.0311 
    f = (a_ex + a_es + a_pol + a_disp + a_dhf) * P * exp_br / 2625.5 - exp_br * (1+br) * q / dr * 0.529177
    return f * m 


@vmap
@jit_condition()
def TT_damping_disp_kernel(dr, m, ex_ai, ex_aj, es_ai, es_aj, pol_ai, pol_aj, disp_ai, disp_aj, dhf_ai, dhf_aj, bi, bj, c6i, c6j, c8i, c8j, c10i, c10j):
    ############################################################
    # include a*exp(-br), q-q, c6-c6, c8-c8, c10-c10 interaction
    # in future, if we predict a, b, c via nn, all of them are 
    # sqrt(a), sqrt(b), sqrt(c), now is sqrt(a), b and sqrt(c)
    ############################################################
    a_ex = (ex_ai * ex_aj)
    a_es = (es_ai * es_aj)
    a_pol = (pol_ai * pol_aj)
    a_disp = (disp_ai * disp_aj)
    a_dhf = (dhf_ai * dhf_aj)
    b = torch.sqrt(bi * bj)
    c6 = c6i * c6j; c8 = c8i * c8j; c10 = c10i * c10j 
    br = b * dr 
    br2 = br * br 
    br3 = br2 * br
    br4 = br3 * br
    br5 = br4 * br
    br6 = br5 * br
    br7 = br6 * br
    br8 = br7 * br
    br9 = br8 * br
    br10 = br9 * br 
    x = br - (2 * br2 + 3 * br) / (br2 + 3 * br + 3) 
    s6 = 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + x**6/720
    s8 = s6 + x**7/5040 + x**8/40320
    s10 = s8 + x**9/362880 + x**10/3628800
    exp_x = torch.exp(-x)
    f6 = exp_x * s6
    f8 = exp_x * s8
    f10 = exp_x * s10
    ###################################################################
    # attention to the unit of sr, damp disp and damp q-q interaction
    # unit of slater-isa is kj/mol; unit of disp is hartree as the 
    # c6 is hartree/A**6 and dr is A; q-q interaction is hartree if r 
    # is bohr, here r is A, so DIELECTRIC is needed to convert kj/mol 
    ####################################################################
    #f = (a_ex + a_es + a_pol + a_disp + a_dhf) * P * exp_br / 2625.5 # 9.3556 
    #f = (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) # 200783 
    #f = - exp_br * (1+br) * q / dr / DIELECTRIC # 0.0311 
    f = (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) 
    return f * m 

@vmap 
def slater_disp_damping_kernel(dr, m, bi, bj, c6i, c6j, c8i, c8j, c10i, c10j):
    r'''
    Slater-ISA type damping for dispersion:
    f(x) = -e^{-x} * \sum_{k} x^k/k!
    x = Br - \frac{2*(Br)^2 + 3Br}{(Br)^2 + 3*Br + 3}
    see jctc 12 3851
    '''
    b = torch.sqrt(bi * bj)
    c6 = c6i * c6j
    c8 = c8i * c8j
    c10 = c10i * c10j
    br = b * dr
    br2 = br * br
    x = br - (2*br2 + 3*br) / (br2 + 3*br + 3)
    s6 = 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + x**6/720
    s8 = s6 + x**7/5040 + x**8/40320
    s10 = s8 + x**9/362880 + x**10/3628800
    exp_x = torch.exp(-x)
    f6 = exp_x * s6
    f8 = exp_x * s8
    f10 = exp_x * s10
    return (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) * m

@vmap 
def slater_sr_kernel(dr, m, ai, aj, bi, bj):
    '''
    Slater-ISA type short range terms
    see jctc 12 3851
    '''
    b = torch.sqrt(bi * bj)
    a = ai * aj
    br = b * dr
    br2 = br * br
    P = 1/3 * br2 + br + 1
    return a * P * torch.exp(-br) * m



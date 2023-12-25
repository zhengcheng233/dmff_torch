from functools import partial
from functorch import vmap
import torch

# This module deals with the transformations and rotations of multipoles
# The important conversion matrices used in multipoles
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
rt3 = 1.73205080757
inv_rt3 = 1.0/rt3
# the dipole conversion matrices, cart2harm and harm2cart
C1_h2c = torch.tensor([[0., 1., 0.],
                    [0., 0., 1.],
                    [1., 0., 0.]], dtype=torch.float32, device=device)
C1_c2h = C1_h2c.T
# the quadrupole conversion matrices
C2_c2h = torch.tensor([[      0.,        0.,     1.,         0.,         0.,         0.],
                    [      0.,        0.,     0.,         0., 2*inv_rt3,         0.],
                    [      0.,        0.,     0.,         0.,         0., 2*inv_rt3],
                    [inv_rt3, -inv_rt3,     0.,         0.,         0.,         0.],
                    [      0.,        0.,     0., 2*inv_rt3,         0.,         0.]], dtype=torch.float32, device=device)

C2_h2c = torch.tensor([[-0.5,     0.,     0.,  rt3/2,     0.],
                    [-0.5,     0.,     0., -rt3/2,     0.],
                    [   1.,     0.,     0.,      0.,     0.],
                    [   0.,     0.,     0.,      0., rt3/2],
                    [   0., rt3/2,     0.,      0.,     0.],
                    [   0.,     0., rt3/2,      0.,     0.]], dtype=torch.float32, device=device)

@partial(vmap, in_dims=(0, 0, None), out_dims=0)
def rot_global2local(Q_gh, localframes, lmax=2):
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
    if lmax > 2:
        raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

    # monopole 
    Q_lh_0 = Q_gh[0:1]
    # for dipole 
    if lmax >= 1:
        # the rotation matrix 
        zxy = torch.tensor([2,0,1], dtype=torch.int32, device=Q_gh.device)
        R1 = localframes[zxy][:,zxy]
        # rotate
        Q_lh_1 = torch.matmul(R1, Q_gh[1:4])
    if lmax >= 2:
        xx = localframes[0, 0]
        xy = localframes[0, 1]
        xz = localframes[0, 2]
        yx = localframes[1, 0]
        yy = localframes[1, 1]
        yz = localframes[1, 2]
        zx = localframes[2, 0]
        zy = localframes[2, 1]
        zz = localframes[2, 2]
        quadrupoles = Q_gh[4:9]
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
        Q_lh_2 = torch.einsum('jk,k->j', C2_gl, quadrupoles)
    if lmax == 0:
        Q_lh = Q_lh_0
    elif lmax == 1:
        Q_lh = torch.hstack([Q_lh_0, Q_lh_1])
    elif lmax == 2:
        Q_lh = torch.hstack([Q_lh_0, Q_lh_1, Q_lh_2])
    return Q_lh

def rot_local2global(Q_lh, localframes, lmax=2):
    '''
    This function rotates harmonic moments Q from global frame to local frame
    Simply use the rot_global2local, and localframe^-1

    Input:
        Q_lh: 
            n * (l+1)^2, stores the local harmonic multipole moments of each site
        localframes: 
            n * 3 * 3, stores the Rotation matrix for each site, the R is defined as:
            [r1, r2, r3]^T, with r1, r2, r3 being the local frame axes
        lmax:
            integer, the maximum multipole order

    Output:
        Q_gh:
            n * (l+1)^2, stores the rotated global harmonic multipole moments
    '''
    return rot_global2local(Q_lh, torch.transpose(localframes, -2, -1), lmax)

@partial(vmap, in_dims=(0, 0), out_dims=0)
def rot_ind_global2local(U_g, localframes,zxy=torch.tensor([2,0,1],dtype=torch.int32, device='cuda:0' if torch.cuda.is_available() else 'cpu')):
    '''
    A special rotation function for just dipoles, aim for applying on induced dipoles
    '''
    R1 = localframes[zxy][:,zxy]
    U_l = torch.matmul(R1, U_g)
    return U_l



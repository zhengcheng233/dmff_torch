import torch
from functools import partial
from functorch import vmap
import numpy as np

def pbc_shift(drvecs, box, box_inv):
    unshifted_dsvecs = torch.matmul(drvecs, box_inv)
    dsvecs = unshifted_dsvecs - torch.floor(unshifted_dsvecs + 0.5)
    return torch.matmul(dsvecs, box)

# direct use the vmap
v_pbc_shift = vmap(pbc_shift, in_dims=(0, None, None), out_dims=0)

def normalize(matrix, axis=1, ord=2):
    '''
    Normalise a matrix along one dimension
    '''
    normalised = matrix / torch.linalg.norm(matrix, axis=axis, keepdims=True, ord=ord)
    return normalised

@partial(vmap, in_dims=(0, 0, 0, 0), out_dims=0)
def build_quasi_internal(r1, r2, dr, norm_dr):
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
    vectorX = torch.where(torch.logical_or(r1[1]!=r2[1],r1[2]!=r2[2]),vectorZ + torch.tensor([1., 0., 0.],dtype=torch.float32, device=r1.device), vectorZ + torch.tensor([0., 1., 0.], dtype=torch.float32, device=r1.device)) 
    dot_xz = torch.matmul(vectorZ, vectorX)
    vectorX = vectorX - vectorZ * dot_xz
    vectorX = vectorX / torch.norm(vectorX)
    vectorY = torch.cross(vectorZ,vectorX)
    return torch.stack([vectorX, vectorY, vectorZ])

def generate_construct_local_frames(axis_types, axis_indices):
    """
    Generates the local frame constructor, common to the same physical system

    inputs:
        axis_types:
            N, a len(N) integer array, labels the types of localframe transformation rules for each atom.
        axis_indices:
            N * 3, indices of z,x,y atoms of the localframe of each atom.

    outputs:
        construct_local_frames:
            function type (positions, box) -> local_frames
    in torch version, only support Z then X, and this is enough
    """
    
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    Zonly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6

    z_atoms = axis_indices[:, 0]
    x_atoms =axis_indices[:, 1]
    y_atoms = axis_indices[:, 2]

    Zonly_filter = (axis_types == Zonly)
    not_Zonly_filter = torch.logical_not(Zonly_filter)
    Bisector_filter = (axis_types == Bisector)
    ZBisect_filter = (axis_types == ZBisect)
    ThreeFold_filter = (axis_types == ThreeFold)

    def construct_local_frames(positions, box):
        '''
        This function constructs the local frames for each site

        Inputs:
            positions:
                N * 3: the positions matrix
            box:
        Outputs:
            #jichen:
            #NOTE: It doesn't seem to return Q
            Q: 
                N*(lmax+1)^2, the multipole moments in global harmonics.
            local_frames:
                N*3*3, the local frames, axes arranged in rows
        '''
        n_sites = positions.shape[0]
        box_inv = torch.linalg.inv(box)

        ### Process the x, y, z vectors according to local axis rules
        vec_z = pbc_shift(positions[z_atoms] - positions, box, box_inv)
        vec_z = normalize(vec_z)
        vec_x = torch.zeros((n_sites, 3), dtype=torch.int32, device=box.device)
        vec_y = torch.zeros((n_sites, 3), dtype=torch.int32, device=box.device)

        # Z-Only
        x_of_vec_z = torch.round(torch.abs(vec_z[:,0]))
        vec_x_Zonly = torch.vstack((1.-x_of_vec_z, x_of_vec_z, torch.zeros_like(x_of_vec_z))).T
        if torch.sum(Zonly_filter) > 0:
            vec_x[Zonly_filter] = (vec_x_Zonly)
        # for those that are not Z-Only, get normalized vecX
        vec_x_not_Zonly = positions[x_atoms[not_Zonly_filter]] - positions[not_Zonly_filter]
        vec_x_not_Zonly = pbc_shift(vec_x_not_Zonly, box, box_inv)
        
        vec_x[not_Zonly_filter] = normalize(vec_x_not_Zonly, axis=1)        
        # Bisector
        if torch.sum(Bisector_filter) > 0:
            vec_z_Bisector = vec_z[Bisector_filter] + vec_x[Bisector_filter]
            vec_z[Bisector_filter] = (normalize(vec_z_Bisector, axis=1))
        # z-bisector
        if torch.sum(ZBisect_filter) > 0:
            vec_y_ZBisect = positions[y_atoms[ZBisect_filter]] - positions[ZBisect_filter]
            vec_y_ZBisect = pbc_shift(vec_y_ZBisect, box, box_inv)
            vec_y_ZBisect = normalize(vec_y_ZBisect, axis=1)
            vec_x_ZBisect = vec_x[ZBisect_filter] + vec_y_ZBisect
            vec_x[ZBisect_filter] = (normalize(vec_x_ZBisect, axis=1))
        # ThreeFold
        if torch.sum(ThreeFold_filter) > 0:
            vec_x_threeFold = vec_x[ThreeFold_filter]
            vec_z_threeFold = vec_z[ThreeFold_filter]

            vec_y_threeFold = positions[y_atoms[ThreeFold_filter]] - positions[ThreeFold_filter]
            vec_y_threeFold = pbc_shift(vec_y_threeFold, box, box_inv)
            vec_y_threeFold = normalize(vec_y_threeFold, axis=1)
            vec_z_threeFold += (vec_x_threeFold + vec_y_threeFold)
            vec_z_threeFold = normalize(vec_z_threeFold)

            vec_y[ThreeFold_filter] = (vec_y_threeFold)
            vec_z[ThreeFold_filter] = (vec_z_threeFold)

        # up to this point, z-axis should already be set up and normalized
        xz_projection = torch.sum(vec_x*vec_z, axis = 1, keepdims=True)
        vec_x = normalize(vec_x - vec_z * xz_projection, axis=1)
        # up to this point, x-axis should be ready
        vec_y = torch.cross(vec_z, vec_x)

        return torch.stack((vec_x, vec_y, vec_z), axis=1)
    return construct_local_frames


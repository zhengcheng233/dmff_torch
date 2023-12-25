#!/usr/bin/env python 
"""
calculate the damping energy 
"""
from dmff_torch.pairwise import (
    TT_damping_qq_c6_kernel,
    generate_pairwise_interaction,
    slater_disp_damping_kernel,
    slater_sr_kernel, 
    TT_damping_qq_kernel
)

def QqTtDamping(positions, box, pairs, mScales, b_list, q_list):
    r"""
    This one calculates the tang-tonnies damping of charge-charge interaction
    E = \sum_ij exp(-B*r)*(1+B*r)*q_i*q_j/r
    """
    pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, b_list, q_list)
    return E_sr 

def SlaterDispDamping(positions, box, pairs, mScales, b_list, c6_list, c8_list, c10_list):
    r"""
    This one computes the slater-type damping function for c6/c8/c10 dispersion
    E = \sum_ij (f6-1)*c6/r6 + (f8-1)*c8/r8 + (f10-1)*c10/r10
    fn = f_tt(x, n)
    x = br - (2*br2 + 3*br) / (br2 + 3*br + 3)
    """
    pot_fn_sr = generate_pairwise_interaction(slater_disp_damping_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, b_list, c6_list,
                             c8_list, c10_list)
    return E_sr

def SlaterDamping(positions, box, pairs, mScales, a_list, b_list):
    r"""
    This one computes the Slater-ISA type exchange interaction
    u = \sum_ij A * (1/3*(Br)^2 + Br + 1)
    """
    pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, a_list, b_list)
    return E_sr

def SlaterSrEs(positions, box, pairs, mScales, a_list, b_list):
    pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, a_list, b_list)
    return E_sr

def SlaterSrPol(positions, box, pairs, mScales, a_list, b_list):
    pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, a_list, b_list)
    return E_sr

def SlaterSrdisp(positions, box, pairs, mScales, a_list, b_list):
    pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, a_list, b_list)
    return E_sr

def SlaterDhf(positions, box, pairs, mScales, a_list, b_list):
    pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
                                                  static_args={})
    E_sr = pot_fn_sr(positions, box, pairs, mScales, a_list, b_list)
    return E_sr

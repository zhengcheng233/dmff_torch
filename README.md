# dmff_torch
Author: Zheng Cheng (chengz@bjaisi.com)

**torch-dmff** (torch-based **D**ifferentiable **M**olecular **F**orce **F**ield) is a PyTorch implementation of the previously JAX-based DMFF([![doi:10.26434/chemrxiv-2022-2c7gv](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2022--2c7gv-blue)](https://doi.org/10.26434/chemrxiv-2022-2c7gv)). It enables the differentiable calculation of the polarizable energy of organics systems whose parameters like atomic multipole moments and dispersion coefficients are predicted via pytorch-based neural networks. 

All interations involved in torch-DMFF are briefly introduced below and the users are encouraged to read the references for more mathematical details:

## 1 Electrostatic Interaction

The electrostatic interaction between two atoms can be described using multipole expansion, in which the electron cloud of an atom can be expanded as a series of multipole moments including charges, dipoles, quadrupoles, and octupoles etc. If only the charges (zero-moment) are considered, it is reduced to the point charge model in classical force fields:

$$
V=\sum_{ij} \frac{q_i q_j}{r_{ij}}
$$

where $q_i$ is the charge of atom $i$.

More complex (and supposedly more accurate) force field can be obtained by including more multipoles with higher orders. Some force fields, such as MPID, goes as high as octupoles. Currently in DMFF, we support up to quadrupoles:

$$
V=\sum_{tu} Q_t^A T^{AB}_{tu} Q_u^B
$$

where $Q_t^A$ represents the t-component of the multipole moment of atom A. Note there are two (equivalent) ways to define multipole moments: cartesian and spherical harmonics. Cartesian representation is over-complete but with a simpler definition, while spherical harmonics are easier to use in real calculations. In the user API, we use cartesian representation, in consistent with the AMOEBA and the MPID plugins in OpenMM. However, spherical harmonics are always used in the computation kernel, and we assume all components are arranged in the following order:

$$0, 10, 1c, 1s, 20, 21c, 21s, 22c, 22s, ...$$

The $T_{tu}^{AB}$ represents the interaction tensor between multipoles. The mathematical expression of these tensors can be found in the appendix F of Ref 1. The user can also find the conversion rule between different representations in Ref 1 & 5.


## 2 Coordinate System for Multipoles

Different to charges, the definition of multipole moments depends on the coordinate system. The exact value of the moment tensor will be rotated in accord to different coordinate systems. There are three types of frames involved in DMFF, each used in a different scenario:

  - Global frame: coordinate system binds to the simulation box. It is same for all the atoms. We use this frame to calculate the charge density structure factor $S(\vec{k})$ in reciprocal space.
  - Local frame: this frame is defined differently on each atom, determined by the positions of its peripheral atoms. Normally, atomic multipole moments are most stable in the local frame, so it is the most suitable frame for force field input. In DMFF API, the local frames are defined using the same way as the AMOEBA plugin in OpenMM. The details can found in the following references:
      * [OpenMM forcefield.py](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/forcefield.py#L4894), line 4894~4933
      * [J. Chem. Theory Comput. 2013, 9, 9, 4046–4063](https://pubs.acs.org/doi/abs/10.1021/ct4003702)
  - Quasi internal frame, aka. QI frame: this frame is defined for each pair of interaction sites, in which the z-axis is pointing from one site to another. In this frame, the real-space interaction tensor ($T_{tu}^{AB}$) can be greatly simplified due to symmetry. We thus use this frame in the real space calculation of PME.


## 3 Polarization Interaction

DMFF supports polarizable force fields, in which the dipole moment of the atom can respond to the change of the external electric field. In practice, each atom has not only permanent multipoles $Q_t$, but also induced dipoles $U_{ind}$. The induced dipole-induced dipole and induced dipole-permanent multipole interactions needs to be damped at short-range to avoid polarization catastrophe. In DMFF, we use the Thole damping scheme identical to MPID (ref 6), which introduces a damping width ($a_i$) for each atom $i$. The damping function is then computed and applied to the corresponding interaction tensor. Taking $U_{ind}$-permanent charge interaction as an example, the definition of damping function is:

$$
\displaylines{
1-\left(1+a u+\frac{1}{2} a^{2} u^{2}\right) e^{-a u} \\ 
a=a_i + a_j \\ 
u=r_{ij}/\left(\alpha_i \alpha_j\right)^{1/6} 
}
$$

Other damping functions between multipole moments can be found in Ref 6, table I. 

It is noted that the atomic damping parameter $a=a_i+a_j$ is only effective on topological neighboring pairs (with $pscale = 0$), while a default value of $a_{default}$ is set for all other pairs. In DMFF, the atomic $a_i$ is specified via the xml API, while $a_{default}$  is controlled by the `dmff.admp.pme.DEFAULT_THOLE_WIDTH` variable, which is set to 5.0 by default.

We solve $U_{ind}$ by minimizing the electrostatic energy:

$$
V=V_{perm-perm}+V_{perm-ind}+V_{ind-ind}
$$

The last two terms are related to $U_{ind}$. Without introducing the nonlinear polarization terms (e.g., some force fields introduce $U^4$ to avoid polarization catastrophe), the last two terms are quadratic to $U_{ind}$: 

$$
V_{perm-ind}+V_{ind-ind}=U^TKU-FU
$$

where the off-diagonal term of $K$ matrix is induced-induced dipole interaction, the diagonal term is formation energy of the induced dipoles ($\sum_i \frac{U_i^2}{2\alpha_i}$); the $F$ matrix represents permanent multipole - induced dipole interaction. We use the gradient descent method to optimize energy to get $U_{ind}$.

In the current version, we temporarily assume that the polarizability is spherically symmetric, thus the polarizability $\alpha_i$ is a scalar, not a tensor. **Thus the inputs (`polarizabilityXX, polarizabilityYY, polarizabilityZZ`) in the xml API is averaged internally**. In future, it is relatively simple to relax this restriction: simply change the reciprocal of the polarizability to the inverse of the matrix when calculating the diagonal terms of the $K$ matrix.

## 4 Dispersion Interaction

In ADMP, we assume that the following expansion is used for the long-range dispersion interaction:

$$
V_{disp}=\sum_{ij}-\frac{C_{ij}^6}{r_{ij}^6}-\frac{C_{ij}^8}{r_{ij}^8}-\frac{C_{ij}^{10}}{r_{ij}^{10}}-...
$$

where the dispersion coefficients are determined by the following combination rule:

$$
C^n_{ij}=\sqrt{C_i^n C_j^n}
$$

Note that the dispersion terms should be consecutive even powers according to the perturbation theory, so the odd dispersion terms are not supported in ADMP. 

In ADMP, this long-range dispersion is computed using PME (*vida infra*), just as electrostatic terms.

In the classical module, dispersions are treated as short-range interactions using standard cutoff scheme.

## 5 Long-Range Interaction with PME

The long-range potential includes electrostatic, polarization, and dispersion (in ADMP) interactions. Taking charge-charge interaction as example, the interaction decays in the form of $O(\frac{1}{r})$, and its energy does not converge with the increase of cutoff distance. The multipole electrostatics and dispersion interactions also converge slow with respect to cutoff distance. We therefore use Particle Meshed Ewald(PME) method to calculate these interactions.

In PME, the interaction tensor is splitted into the short-range part and the long-range part, which are tackled in real space and reciprocal space, respectively. For example, the Coulomb interaction is decomposed as:

$$
\frac{1}{r}=\frac{erfc(\kappa r)}{r}+\frac{erf(\kappa r)}{r}
$$

The first term is a short-range term, which can be calculated directly by using a simple distance cutoff in real space. The second term is a long-range term, which needs to be calculated in reciprocal space by fast Fourier transform(FFT). The total energy of charge-charge interaction is computed as:

$$
\displaylines{
E_{real} = \sum_{ij}\frac{erfc(\kappa r_{ij})}{r_{ij}}  \\
E_{recip} = \sum_{\vec{k}\neq 0} {\frac{2\pi}{Vk^2}\exp\left[\frac{k^2}{4\kappa^2}\right]\left|S(\vec{k})\right|^2}\frac{1}{\left|\theta(\vec{k})\right|^2} \\ 
E_{self} = -\frac{\kappa}{\sqrt{\pi}}\sum_i {q_i^2} \\
E = E_{real}+E_{recip}+E_{self}
}
$$

As for multipolar PME and dispersion PME, the users and developers are referred to Ref 2, 3, and 5 for mathematical details.

The key parameters in PME include:

  - $\kappa$: controls the separation of the long-range and the short-range. The larger $\kappa$ is, the faster the real space energy decays, the smaller the cutoff distance can be used in the real space, and more difficult it is to converge the reciprocal energy and the larger $K_{max}$ it needs;

  - $r_{c}$: cutoff distance in real space;

  - $K_{max}$: controls the number of maximum k-points in all three dimensions


In DMFF, we determine these parameters in the same way as in [OpenMM](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald):

$$
\displaylines{
\kappa=\sqrt{-\log (2 \delta)} / r_{c} \\ 
K_{max}=\frac{2 \kappa d}{3 d^{1 / 5}}
}
$$

where the user needs to specify the cutoff distance $r_c$ when building the neighbor list, the width of the box in each dimension $d$ (determined from the input box matrix), and the energy accuracy $\delta$.

In the current version, the dispersion PME calculator uses the same parameters as in electrostatic PME.

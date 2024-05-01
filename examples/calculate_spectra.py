#!/usr/bin/env python
# 
# This example is meant to show a complete walkthrough for computing
# the dirichlet and neumann modes for the wind-driven gyre example from
# L. Thiry's MQGeometry.
#
# Once the sparse matrices are created with this script, the dirichlet
# and neumann mode eigenpairs can be diagnosed with ../bin/laplacian_modes
#
# From here, the eigenmodes and eigenvalues can be used to calcualte the spectra 
# of the velocity field obtained with a QG simulation from MQGeometry.
# 
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from fluidspectraig.splig import splig
from fluidspectraig.nma import NMA
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.switch_backend('agg')


#case_dir = "mqgeometry_doublegyre-squarebasin/16x16"
#case_dir = "mqgeometry_doublegyre-octagon/16x16/"
case_dir = "square_domain/64x64/"

nma_obj = NMA()
nma_obj.load(case_dir)

u = torch.ones((nma_obj.splig_d.nx,nma_obj.splig_n.ny),**nma_obj.arr_kwargs)
v = torch.zeros((nma_obj.splig_n.nx,nma_obj.splig_d.ny),**nma_obj.arr_kwargs)

print(f"shape(u) : {u.shape}")
print(f"shape(v) : {v.shape}")

#nma_obj.plot_eigenmodes()


# lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m = nma_obj.spectra(u,v)

# print(f" Estimated energy (m^2/s^2): {np.sum(Edb_m)}")
# plt.figure
# # neumann mode - divergent component
# plt.plot( lambda_m, Edi_m, label="d_i", marker="o" )
# plt.plot( lambda_m, Edb_m, label="d_b", marker="o" )
# # dirichlet mode - rotational component
# plt.plot( sigma_m, Eri_m, label="r_i", marker="o" )
# plt.plot( sigma_m, Erb_m, label="r_b", marker="o" )
# plt.title("spectra")
# plt.xlabel("$\lambda")
# plt.ylabel("E")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# plt.savefig(f"{case_dir}/constant_spectra.png")


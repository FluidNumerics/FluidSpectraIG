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


#case_dir = "mqgeometry_doublegyre-squarebasin/16x16"
case_dir = "mqgeometry_doublegyre-octagon/16x16/"

model = NMA()
model.load(case_dir)

u = torch.ones((model.splig_d.nx,model.splig_n.ny),**model.arr_kwargs)
v = torch.zeros((model.splig_n.nx,model.splig_d.ny),**model.arr_kwargs)

print(f" shape(u) : {u.shape}")
print(f" shape(v) : {v.shape}")
print("")

#model.plot_eigenmodes()


lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m = model.spectra(u,v)

plt.figure
# neumann mode - divergent component
plt.plot( lambda_m, Edi_m, label="d_i", marker="o" )
plt.plot( lambda_m, Edb_m, label="d_b", marker="o" )
# dirichlet mode - rotational component
plt.plot( sigma_m, Eri_m, label="r_i", marker="o" )
plt.plot( sigma_m, Erb_m, label="r_b", marker="o" )
plt.savefig(f"{case_dir}/constant_spectra.png")


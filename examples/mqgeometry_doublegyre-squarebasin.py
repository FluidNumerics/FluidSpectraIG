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
from fluidspectraig.tuml import TUML
from fluidspectraig.splig import splig
import os

nx = 1024
ny = 1024

output_dir = f"mqgeometry_doublegyre-squarebasin/{nx}x{ny}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the model
param = {'nx': nx,
         'ny': ny,
         'Lx': 5120.0, # units are in km
         'Ly': 5120.0, # units are in km
         'device': 'cpu',
         'dtype': torch.float64}

model = TUML(param)

# Getting the dirichlet mode mask, grid, and laplacian operator.
mask_g = model.masks.psi.type(torch.int32).squeeze().cpu().numpy()
xg = model.xg.cpu().numpy()
yg = model.yg.cpu().numpy()
dirichlet_matrix_action = model.apply_laplacian_g

mask_c = model.masks.q.type(torch.int32).squeeze().cpu().numpy()
xc = model.xc.cpu().numpy()
yc = model.yc.cpu().numpy()
neumann_matrix_action = model.apply_laplacian_c

print(f"------------------------------")
print(f"Building dirichlet mode matrix")
print(f"------------------------------")
Ld = splig(xg,yg,mask_g,dirichlet_matrix_action) # Dirichlet mode 
print(f"")
print(f"----------------------------")
print(f"Building neumann mode matrix")
print(f"----------------------------")
Ln = splig(xc,yc,mask_c,neumann_matrix_action) # Neumann mode 
print(f"")

# Write structures to file
filename = f"{output_dir}/dirichlet"
Ld.write(filename)

filename = f"{output_dir}/neumann"
Ln.write(filename)

# Save a few plots for reference
# Plot the mask
plt.figure()
plt.imshow(mask_g,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{output_dir}/dirichlet-mask.png')
plt.close()

plt.figure()
plt.imshow(mask_c,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{output_dir}/neumann-mask.png')
plt.close()


#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from fluidspectraig.tuml import TUML
from fluidspectraig.splig import splig
import os

nx = 1024
ny = 1024

output_dir = f"mqgeometry_doublegyre-octagon/{nx}x{ny}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the model
param = {'nx': nx,
         'ny': ny,
         'Lx': 5120.0, # units are in km
         'Ly': 5120.0, # units are in km
         'device': 'cpu',
         'dtype': torch.float64}

mask = torch.ones(nx, ny)
for i in range(nx//4):
    for j in range(ny//4):
        if i+j < min(nx//4, ny//4):
            mask[i,j] = 0.
            mask[i,-1-j] = 0.
            mask[-1-i,j] = 0.
            mask[-1-i,-1-j] = 0.
param['mask'] = mask

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


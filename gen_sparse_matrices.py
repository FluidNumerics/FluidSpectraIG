#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
import numpy as np
import numpy.ma as ma

class splig:
    """ Sparse Laplacian - Irregular Geometry """
    def __init__(self, mask, action=None):
 
      nx, ny = mask.shape
      self.nx = nx
      self.ny = ny

      # Create array to get matrix row from i,j grid indices
      self.matrix_row = ma.array( np.ones((nx,ny)), dtype=np.int32, order='C', mask=np.abs(mask-1),fill_value=-1 ).cumsum().reshape((nx,ny))-1

      # Total number of degrees of freedom
      self.ndof = self.matrix_row.count()

      # Create arrays to map from matrix row to i,j indices
      indices = np.indices((nx,ny))
      self.i_indices = ma.array(indices[0,:,:].squeeze(), dtype=np.int32, order='C', mask=np.abs(mask-1) ).compressed()
      self.j_indices = ma.array(indices[1,:,:].squeeze(), dtype=np.int32, order='C', mask=np.abs(mask-1) ).compressed()

      print(f"Bounding domain shape (nx,ny) : {nx}, {ny}")
      print(f"Number of degrees of freedom : {self.ndof}")

      self.matrix_action = action


# Create the model
param = {'nx': 16,
         'ny': 16,
         'Lx': 5120.0e3,
         'Ly': 5120.0e3,
         'device': 'cuda',
         'dtype': torch.float64}

nx = param['nx']
ny = param['ny']
# mask = torch.ones(nx, ny)
# for i in range(nx//4):
#     for j in range(ny//4):
#         if i+j < min(nx//4, ny//4):
#             mask[i,j] = 0.
#             mask[i,-1-j] = 0.
#             mask[-1-i,j] = 0.
#             mask[-1-i,-1-j] = 0.
# param['mask'] = mask

model = NMA(param)

# Getting the dirichlet mode mask
mask = model.masks.psi.type(torch.int32).squeeze().cpu().numpy()

plt.figure()
plt.imshow(mask,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig('mask.png')
plt.close()

laplacian = splig(mask)

plt.figure()
plt.pcolor(laplacian.matrix_row)
plt.colorbar(fraction=0.046,location='right')
plt.savefig('matrix_row.png')
plt.close()

# Generate impulse fields
impulse = torch.zeros(sum(((4,4), model.xg.shape), ()),**model.arr_kwargs)
print(impulse.shape)

indices = np.indices(model.xg.shape)
for j in np.arange(4):
    j_index = indices[1,:,:]-j
    for i in np.arange(4):
        i_index = indices[0,:,:]-i

        imp = (i_index % 4) + (j_index % 4) == 0
        impulse[i,j,imp] = 1.0

        plt.figure()  
        A = impulse[i,j,:,:].squeeze().cpu().numpy()
        plt.imshow(A,interpolation='nearest', aspect='equal')
        plt.colorbar(fraction=0.046,location='right')
        plt.savefig(f'impulse-{i}-{j}.png')
        plt.close()

# Compute impulse response
irf = -model.apply_laplacian_g(impulse)
for j in np.arange(4):
    for i in np.arange(4):
        plt.figure()  
        A = irf[i,j,:,:].squeeze().cpu().numpy()
        plt.imshow(A,interpolation='nearest', aspect='equal')
        plt.colorbar(fraction=0.046,location='right')
        plt.savefig(f'irf-{i}-{j}.png')
        plt.close()


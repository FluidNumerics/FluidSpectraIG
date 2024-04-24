#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
import numpy as np
import numpy.ma as ma

from petsc4py import PETSc


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

filename_base = "squaredomain"
# Create the model
param = {'nx': 16,
         'ny': 16,
         'Lx': 16.0,
         'Ly': 16.0,
         'device': 'cuda',
         'dtype': torch.float64}

# param = {'nx': 16,
#          'ny': 16,
#          'Lx': 5120.0e3,
#          'Ly': 5120.0e3,
#          'device': 'cuda',
#          'dtype': torch.float64}

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


# For each matrix row, we need to compute the number of non-zero entries
# Each row has at least one non-zero value
nnz = np.ones(laplacian.ndof, dtype=np.int32)

for r in np.arange(laplacian.ndof):
    # Get i,j for this row
    i = laplacian.i_indices[r]
    j = laplacian.j_indices[r]

    # Check south neighbor
    k = laplacian.matrix_row[i,j-1]
    if k >= 0:
        nnz[r] += 1

    # Check north neighbor
    k = laplacian.matrix_row[i,j+1]
    if k >= 0:
        nnz[r] += 1

    # Check west neighbor
    k = laplacian.matrix_row[i-1,j]
    if k >= 0:
        nnz[r] += 1

    # Check east neighbor
    k = laplacian.matrix_row[i+1,j]
    if k >= 0:
        nnz[r] += 1

print(f' Max(nnz) : {max(nnz)}')
print(f' Min(nnz) : {min(nnz)}')

# Now that we have the impulse and impulse response fields, we can look at creating the sparse matrix
# Followed https://tbetcke.github.io/hpc_lecture_notes/petsc_for_sparse_systems.html as a guide
A = PETSc.Mat()

# https://petsc.org/release/manualpages/Mat/MatCreateSeqAIJ/
# nnz = array containing the number of nonzeros in the various rows (possibly different for each row) or NULL
A.createAIJ([laplacian.ndof, laplacian.ndof], nnz=nnz)

# Now we can set the values
irf_cpu = irf.squeeze().cpu().numpy()

interior_template = mask == 1
for j_shift in np.arange(4):
    j_index = indices[1,:,:]-j_shift
    for i_shift in np.arange(4):
        i_index = indices[0,:,:]-i_shift

        # Get the indices for the impulses
        # This gives us a 2-d grid function that is true at
        # each impulse location in the interior of the domain.
        imp = ( ( (i_index % 4) + (j_index % 4) == 0 )*interior_template )
        impulse_indices = [(i,j) for i, row in enumerate(imp) for j, entry in enumerate(row) if entry]

        for i,j in impulse_indices:
            # For each impulse, we get the dof index for the laplacian stencil points
            # and fill the matrix

            # Get the central point of the stencil (diagonal)
            row = laplacian.matrix_row[i,j]
            A.setValue(row,row,irf_cpu[i_shift,j_shift,i,j])

            # Check south neighbor
            col = laplacian.matrix_row[i,j-1]
            if col >= 0:
                A.setValue(row,col,irf_cpu[i_shift,j_shift,i,j-1])

            # Check north neighbor
            col = laplacian.matrix_row[i,j+1]
            if col >= 0:
                A.setValue(row,col,irf_cpu[i_shift,j_shift,i,j+1])

            # Check west neighbor
            col = laplacian.matrix_row[i-1,j]
            if col >= 0:
                A.setValue(row,col,irf_cpu[i_shift,j_shift,i-1,j])

            # Check east neighbor
            col = laplacian.matrix_row[i+1,j]
            if col >= 0:
                A.setValue(row,col,irf_cpu[i_shift,j_shift,i+1,j])

# Assemble the matrix
A.assemble()       
print(f"Matrix Size : {A.size}")
print(f"Matrix is symmetric : {A.isSymmetric()}")
print(A.getInfo())

# Write matrix to file
filename = f"{filename_base}-{nx}-{ny}_dirichlet.dat"
viewer = PETSc.Viewer().createBinary(filename, 'w')
viewer(A)




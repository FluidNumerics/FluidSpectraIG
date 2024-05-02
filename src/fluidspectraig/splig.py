#!/usr/bin/env python

# Define the class for a sparse Laplacian with irregular geometry
# This class can be used to build a sparse matrix for a Laplacian operator
# in 2-D. We assume the irregular geometry is managed via a mask that is
# set to 1 on "wet/interior" points and 0 on "dry/exterior" points.
# The user provides a matrix action routine.
#
# The sparse matrix is built by passing impulse functions into the matrix
# action routine to obtain impulse response functions. From the impulse
# and impulse response function information, we construct a PETSc sparse
# matrix that can be written to file.
#
# When generating the impulse functions and diagnosing matrix entries from
# the IRF, we assume that the Laplacian is the centered 5-point stencil.
# No other stencil options are available at this time.

import torch
import numpy as np
import numpy.ma as ma
from petsc4py import PETSc

def splig_load(filename_base):
    """
    Loads splig object attributes from numpy file {filename_base}.npz 
    Only the grid, mask, and mapping attributes (matrix_row and i/j_indices)
    are loaded. The intention is to use this information for processing
    eigenvectors/eigenvalues from SLEPC.
    """

    filename = f"{filename_base}.npz"
    print(f"Loading SPLIG from {filename}")
    with np.load(filename) as data:
         return splig( mask = data['mask'] )

class splig:
    """ Sparse Laplacian - Irregular Geometry """
    def __init__(self, mask, action=None, dtype=torch.float64, device='cpu' ):
 
      nx, ny = mask.shape
      self.nx = nx
      self.ny = ny

      self.dtype = dtype
      self.device = device
      self.arr_kwargs = {'dtype':self.dtype, 'device': self.device}
      self.mask = mask
      self.matrix_action = action

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

      if action == None:
        self.impulse = None
        self.irf = None
      else:
        self.generate_impulse_fields()
        print(f"Applying matrix action ({self.matrix_action}) to obtain impulse response function")
        self.irf = self.matrix_action(self.impulse)
        self.create_petsc_matrix()

    def write(self,filename_base):
        """
        Writes the PETSc matrix to PETSc binary file at {filename_base}.dat
        and writes the other splig object attributes to numpy file {filename_base}.npz 
        """

        # Write matrix to file
        filename = f"{filename_base}.dat"
        viewer = PETSc.Viewer().createBinary(filename, 'w')
        viewer(self.matrix)

        # Write the grid, mask, impulse fields, impulse response fields, and other object attributes to np file.
        filename = f"{filename_base}.npz"
        np.savez(filename, mask=self.mask,
            matrix_row=self.matrix_row,ndof=self.ndof,
            i_indices=self.i_indices,j_indices=self.j_indices,
            impulse=self.impulse.cpu().numpy(),irf=self.irf.cpu().numpy())

    def generate_impulse_fields(self):

      print(f"Generating impulse fields")
      # Create the impulse fields
      self.impulse = torch.zeros((4,4,self.nx,self.ny),**self.arr_kwargs)

      indices = np.indices((self.nx,self.ny))
      for j in np.arange(4):
        j_index = indices[1,:,:]-j
        for i in np.arange(4):
            i_index = indices[0,:,:]-i

            imp = (i_index % 4) + (j_index % 4) == 0
            self.impulse[i,j,imp] = 1.0

    def create_petsc_matrix(self):

        nnz = np.ones(self.ndof, dtype=np.int32)

        for r in np.arange(self.ndof):
            # Get i,j for this row
            i = self.i_indices[r]
            j = self.j_indices[r]

            # Check south neighbor
            if j > 0:
                k = self.matrix_row[i,j-1]
                if k >= 0:
                    nnz[r] += 1

            # Check north neighbor
            if j < self.ny-1:
                k = self.matrix_row[i,j+1]
                if k >= 0:
                    nnz[r] += 1

            # Check west neighbor
            if i > 0:
                k = self.matrix_row[i-1,j]
                if k >= 0:
                    nnz[r] += 1

            # Check east neighbor
            if i < self.nx-1:
                k = self.matrix_row[i+1,j]
                if k >= 0:
                    nnz[r] += 1

        # Now that we have the impulse and impulse response fields, we can look at creating the sparse matrix
        # Followed https://tbetcke.github.io/hpc_lecture_notes/petsc_for_sparse_systems.html as a guide
        self.matrix = PETSc.Mat()

        # https://petsc.org/release/manualpages/Mat/MatCreateSeqAIJ/
        # nnz = array containing the number of nonzeros in the various rows (possibly different for each row) or NULL
        self.matrix.createAIJ([self.ndof, self.ndof], nnz=nnz)

        # Now we can set the values
        irf = self.irf.squeeze().cpu().numpy()
        indices = np.indices((self.nx,self.ny))
        interior_template = self.mask == 1
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
                    # For each impulse, we get the dof index for the Laplacian stencil points
                    # and fill the matrix

                    # Get the central point of the stencil (diagonal)
                    row = self.matrix_row[i,j]
                    self.matrix.setValue(row,row,irf[i_shift,j_shift,i,j])

                    # Check south neighbor
                    if j > 0:
                        col = self.matrix_row[i,j-1]
                        if col >= 0:
                            self.matrix.setValue(row,col,irf[i_shift,j_shift,i,j-1])

                    # Check north neighbor
                    if j < self.ny-1:
                        col = self.matrix_row[i,j+1]
                        if col >= 0:
                            self.matrix.setValue(row,col,irf[i_shift,j_shift,i,j+1])

                    # Check west neighbor
                    if i > 0:
                        col = self.matrix_row[i-1,j]
                        if col >= 0:
                            self.matrix.setValue(row,col,irf[i_shift,j_shift,i-1,j])

                    # Check east neighbor
                    if i < self.nx-1:
                        col = self.matrix_row[i+1,j]
                        if col >= 0:
                            self.matrix.setValue(row,col,irf[i_shift,j_shift,i+1,j])

        # Assemble the matrix
        self.matrix.assemble()       
        print(f"Matrix Size : {self.matrix.size}")
        print(f"Matrix is symmetric : {self.matrix.isSymmetric()}")
        print(f"Matrix information: {self.matrix.getInfo()}")



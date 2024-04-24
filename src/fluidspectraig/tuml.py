#!/usr/bin/env python
#

# Notes
#
# Assumptions
# ---------------
#    Uniform grid spacing in each direction ( dx = constant, dy = constant )
#
#    The dirichlet modes are associated with the rotational part of the flow (stream function)
#    and are defined on the vorticity points.
# 
#    The neumann modes are associated with the divergent part of the flow (velocity potential)
#    and are defined on the tracer points    
#
#
# Grid
# -------
#   The mask applies to vorticity points on the
#   arakawa c-grid (z-points below).
#
#
#   Vorticity points are in the range (0:nx,0:ny) [nx+1, ny+1] grid points
#   Boundary values for the vorticity 
#
#   Vorticity points are suffixed with "g", e.g. "xg" and "yg" refer to
#   zonal and meridional positions at vorticity points   
#
#   Tracer points have a range of (0,nx-1,0:ny-1) [nx,ny] grid points
#   
#   Tracer points are suffixed with "c", e.g. "xc" and "yc" refer to
#   zonal and meridional positions at tracer points   
#
#   Tracer points are defined offset to the north and east from a vorticity
#   point by half the tracer cell width with the same (i,j) index. 
#   For example, xc(i,j) = xg(i,j) + dx*0.5 and yc(i,j) = yg(i,j) + dy*0.5
#
#
#     z(i,j+1) ----- z(i+1,j+1)
#       |                 |
#       |                 |
#       |                 |
#       |      t(i,j)     |
#       |                 |
#       |                 |
#       |                 |
#     z(i,j) -------- z(i+1,j)
#
#
# Masking
# ------------
#   A mask value of 0 corresponds to a wet cell (this cell is not masked)
#   A mask value of 1 corresponds to a dry cell (this cell is masked)
#   This helps with working with numpy's masked arrays
#


import numpy as np
import torch
from fluidspectraig.masks import Masks
from fluidspectraig.elliptic import laplacian_c, laplacian_g


class TUML:
    """Torch Uniform Mesh Laplacian"""
    def __init__(self, param):
        self.nx = param['nx']
        self.Lx = param['Lx']
        self.ny = param['ny']
        self.Ly = param['Ly']
        self.device = param['device']
        self.dtype = torch.float64
            
        self.arr_kwargs = {'dtype':self.dtype, 'device': self.device}

        # grid
        self.xg, self.yg = torch.meshgrid(torch.linspace(0, self.Lx, self.nx+1, **self.arr_kwargs),
                                        torch.linspace(0, self.Ly, self.ny+1, **self.arr_kwargs),
                                        indexing='ij')

        self.dx = torch.tensor(self.Lx / self.nx, **self.arr_kwargs)
        self.dy = torch.tensor(self.Ly / self.ny, **self.arr_kwargs)

        self.xc, self.yc = torch.meshgrid(torch.linspace(self.dx*0.5, self.Lx-self.dx*0.5, self.nx, **self.arr_kwargs),
                                        torch.linspace(self.dy*0.5, self.Ly-self.dy*0.5, self.ny, **self.arr_kwargs),
                                        indexing='ij')

        mask = param['mask'] if 'mask' in param.keys()  else torch.ones(self.nx, self.ny)
        self.masks = Masks(mask.type(self.dtype).to(self.device))

        # precompile torch functions
        comp =  torch.__version__[0] == '2'
        self.laplacian_g = torch.compile(laplacian_g) if comp else laplacian_g
        self.laplacian_c = torch.compile(laplacian_c) if comp else laplacian_c

        if not comp:
            print('Need torch >= 2.0 to use torch.compile, current version '
                 f'{torch.__version__}, the solver will be slower! ')


    def apply_laplacian_c(self,x):
        return -self.laplacian_c(x,self.masks.u,self.masks.v,self.dx,self.dy)*self.masks.q.squeeze()


    def apply_laplacian_g(self,f):
        fm_g = self.masks.psi.squeeze()*f # Mask the data to apply homogeneous dirichlet boundary conditions
        return -self.masks.psi.squeeze()*self.laplacian_g(fm_g,self.masks.psi,self.dx,self.dy)

    def model_area_g(self):
        return torch.sum( self.masks.psi*self.dx*self.dy ) 
    
    def model_area_c(self):
        return torch.sum( self.masks.q*self.dx*self.dy ) 


        # Compute the shifts. For each type (dirichlet and neumann)
        # we want to have "nclusters" of eigenmodes that are centered
        # around the shifts.
        #
        # each shift is computed as 
        #
        #   ds = (e_high - e_low)/nclusters
        #   shift = e_low + ds*(i+1/2); i = [0,nclusters)
        #
        # where "e_high" is the highest estimated eigenvalue
        # and "e_low" is the lowest estimated eigenvalue
        #
        #  For each shift (dirichlet)
        #     compute eigenvalues, eigenvectors in the neighborhood of shift
        #
        #  For each shift (neumann)
        #     compute eigenvalues, eigenvectors in the neighborhood of shift
        #
        #
        #  Verify orthogonality
        #
        #
        #  save eigenvalues and eigenvectors to disk
        #
    # def findEigenmodes(self, nmodes=10, tolerance=0, deShift=0, neShift=1e-2):
    #     """Finds the eigenmodes using sci-py `eigsh`.

    #     Parameters

    #     nmodes - the number of eigenmodes you wish to find
    #     sigma  - Eigenmodes with eigenvalues near sigma are returned
    #     which  - Identical to the scipy `which` argument
    #     deShift - factor to shift the diagonal by for the dirichlet mode operator
    #     neShift - factor to shift the diagonal by for the neumann mode operator

    #     See scipy/eigsh docs for more details

    #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

    #     """
        
    #     self.findDirichletModes(nmodes, tolerance, deShift)
    #     self.findNeumannModes(nmodes, tolerance, neShift)

    # def vectorProjection(self, u, v):
    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels

    #     # if the eigenmodes have not been found
    #     # find them, using the default parameters
    #     # if (self.eigenmodes is None) :
    #     #     self.findEigenmodes()

    #     print("Calculating projection of u,v")

    #     # Calculate the divergence
    #     divergence = (
    #         kernels.divergence(
    #             u, v, self.dxg, self.dyg, self.hFacW, self.hFacS, self.rac
    #         )
    #         * self.maskC
    #     )

    #     # Calculate the vorticity
    #     vorticity = kernels.vorticity(u, v, self.dxc, self.dyc, self.raz) * self.maskZ

    #     nmodes = self.d_eigenvalues.shape[0]
    #     ny, nx = u.shape
    #     db_m = np.zeros(
    #         (nmodes), dtype=self.prec
    #     )  # Projection of divergence onto the neumann modes (boundary)
    #     di_m = np.zeros(
    #         (nmodes), dtype=self.prec
    #     )  # Projection of divergence onto the neumann modes (interior)
    #     vb_m = np.zeros(
    #         (nmodes), dtype=self.prec
    #     ) # Projection of vorticity onto the dirichlet modes (boundary)
    #     vi_m = np.zeros(
    #         (nmodes), dtype=self.prec
    #     )  # Projection of vorticity onto the dirichlet modes (interior)

    #     for k in range(0, nmodes):
    #         vi_m[k] = np.sum(
    #             vorticity * np.squeeze(self.d_eigenmodes[k, :, :]) * self.raz
    #         )  # Projection of vorticity onto the dirichlet modes
    #         di_m[k] = np.sum(
    #             divergence * np.squeeze(self.n_eigenmodes[k, :, :]) * self.rac
    #         )  # Projection of divergence onto the neumann modes

    #         # Calculate the n_eigenmodes on u-points
    #         etak = np.squeeze(self.n_eigenmodes[k, :, :])
    #         etak = kernels.prolongTracer(etak)
    #         uEtak = kernels.TtoU(etak) * u
    #         # Calculate the n_eigenmodes on v-points
    #         vEtak = kernels.TtoV(etak) * v

    #         # Calculate the divergence of \vec{u} \eta_k
    #         divUEta = (
    #             kernels.divergence(
    #                 uEtak, vEtak, self.dxg, self.dyg, self.hFacW, self.hFacS, self.rac
    #             )
    #             * self.maskC
    #         )
    #         # Subtract the boundary contribution from the divergence coefficients
    #         db_m[k] = -np.sum(divUEta * self.rac)

    #     return di_m, db_m, vi_m, vb_m
    
    # def spectra(self, u, v, decimals=8):
    #     """Calculates the energy spectra for a velocity field (u,v).
        
    #     This routine calls the vectorProjection routine to obtain spectral
    #     coefficients :

    #         di_m - Divergent (Neumann) mode projection coefficients, interior component
    #         db_m - Dirichlet (Neumann) mode projection coefficients, boundary component
    #         vi_m - Vorticity (Dirichlet) mode projection coefficients, interior component
    #         vb_m - Vorticity (Dirichlet) mode projection coefficients, interior component

    #     The energy is broken down into four parts

    #         1. Divergent interior
    #         2. Rotational interior
    #         3. Divergent boundary
    #         4. Rotational boundary
        
    #     Each component is defined as

    #         1. Edi_{m} = -0.5*di_m*di_m/\lambda_m 
    #         2. Eri_{m} = -0.5*vi_m*vi_m/\sigma_m 
    #         3. Edb_{m} = -(0.5*db_m*db_m + db_m*di_m)/\lambda_m 
    #         4. Erb_{m} = -(0.5*vb_m*vb_m + vb_m*vi_m)/\sigma_m         

    #     Once calculated, the spectra is constructed as four components

    #         1. { \lambda_m, Edi_m }_{m=0}^{N}
    #         2. { \sigma_m, Eri_m }_{m=0}^{N}
    #         3. { \lambda_m, Edb_m }_{m=0}^{N}
    #         4. { \sigma_m, Erb_m }_{m=0}^{N}
 
    #     Energy associated with degenerate eigenmodes are accumulated to a single value. Eigenmodes are deemed
    #     "degenerate" if their eigenvalues similar out to "decimals" decimal places. The eigenvalue chosen
    #     for the purpose of the spectra is the average of the eigenvalues of the degenerate modes.
        
    #     """

    #     # Calculate the spectral coefficients
    #     di_m, db_m, vi_m, vb_m = self.vectorProjection(u,v)

    #     # Calculate the energy associated with interior vorticity
    #     Edi = -0.5 * di_m * di_m / self.n_eigenvalues
    #     Edi[self.n_eigenvalues == 0.0] = 0.0

    #     # Calculate the energy associated with boundary vorticity
    #     Edb = -(0.5 * db_m * db_m + di_m*db_m) / self.n_eigenvalues
    #     Edb[self.n_eigenvalues == 0.0] = 0.0

    #     # Calculate the energy associated with interior vorticity
    #     Eri = -0.5 * vi_m * vi_m / self.d_eigenvalues

    #     # Calculate the energy associated with boundary vorticity
    #     Erb = -(0.5 * vb_m * vb_m + vi_m*vb_m) / self.d_eigenvalues

    #     n_evals_rounded = np.round(self.n_eigenvalues,decimals=decimals)
    #     # Collapse degenerate modes
    #     lambda_m = np.unique(n_evals_rounded)
    #     Edi_m = np.zeros_like(lambda_m)
    #     Edb_m = np.zeros_like(lambda_m)
    #     k = 0
    #     for ev in lambda_m:
    #         Edi_m[k] = np.sum(Edi[n_evals_rounded == ev])
    #         Edb_m[k] = np.sum(Edb[n_evals_rounded == ev])
    #         k+=1

    #     d_evals_rounded = np.round(self.d_eigenvalues,decimals=decimals) 
    #     sigma_m = np.unique(d_evals_rounded)
    #     Eri_m = np.zeros_like(sigma_m)
    #     Erb_m = np.zeros_like(sigma_m)
    #     k = 0
    #     for ev in sigma_m:
    #         Eri_m[k] = np.sum(Eri[d_evals_rounded == ev])
    #         Erb_m[k] = np.sum(Erb[d_evals_rounded == ev])
    #         k+=1

    #     return lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m
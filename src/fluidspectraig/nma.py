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
import torch.nn.functional as F

from fluidspectraig.fd import grad_perp, interp_TP, laplacian_h
from fluidspectraig.helmholtz import compute_laplace_dst, solve_helmholtz_dst, \
                      solve_helmholtz_dst_cmm, compute_dst_capacitance_matrices, \
                      compute_laplace_dct, solve_helmholtz_dct
from fluidspectraig.masks import Masks
from fluidspectraig.mfeigen_torch import implicitly_restarted_lanczos,norm, dot
from fluidspectraig.mfpcg_torch import pcg

zeroTol = 1e-12

def dfdx_c(f,dx):
    """Calculates the x-derivative of a function on tracer points
    and returns a function on u-points.Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,1:,:]-f[...,:-1,:])/dx, (0,0,1,1), mode='constant',value=0.
    )

def dfdx_u(f,dx):
    """Calculates the x-derivative of a function on u points
    and returns a function on tracer-points."""
    return (f[...,1:,:]-f[...,:-1,:])/dx

def dfdy_c(f,dy):
    """Calculates the y-derivative of a function on tracer points
    and returns a function on v-points. Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,:,1:]-f[...,:,:-1])/dy, (1,1,0,0), mode='constant',value=0.
    )

def dfdy_v(f,dy):
    """Calculates the y-derivative of a function on v points
    and returns a function on tracer points."""
    return (f[...,:,1:]-f[...,:,:-1])/dy

def laplacian_c(f, masku, maskv, shift, dx, dy):
    """2-D laplacian on the tracer points. On tracer points, we are
    working with the divergent modes, which are associated with neumann
    boundary conditions. """
    return dfdx_u( dfdx_c(f,dx)*masku, dx ) + dfdy_v( dfdy_c(f,dy)*maskv, dy ) - shift*f

def laplacian_g(f, maskz, shift, dx, dy):
    """2-D laplacian on the vorticity points. On vorticity points, we are
    working with the rotational modes, which are associated with dirichlet 
    boundary conditions. Function values are assumed to be masked prior
    to calling this method. Additionally, the laplacian is returned
    as zero numerical boundaries"""
    # function adapted from github.com/louity/MQGeometry
    # Copyright (c) 2023 louity
    return F.pad(
        (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1]) / dx**2 \
      + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1]) / dy**2,
        (1,1,1,1), mode='constant', value=0.)*maskz - shift*f

class NMA:
    def __init__(self, param):
        self.nx = param['nx']
        self.Lx = param['Lx']
        self.ny = param['ny']
        self.Ly = param['Ly']
        self.nmodes = param['nmodes']
        self.nkrylov = param['nkrylov']
        self.device = param['device']
        self.dtype = torch.float64
        self.pcg_tol = 1e-10
        self.pcg_max_iter = 1500
        self.arr_kwargs = {'dtype':self.dtype, 'device': self.device}

        self.n_neumann = 0
        self.n_dirichlet = 0
        self.n_shift = 1e-6
        self.d_shift = 0.0

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

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # precompile torch functions
        comp =  torch.__version__[0] == '2'
        self.laplacian_g = torch.compile(laplacian_g) if comp else laplacian_g
        self.laplacian_c = torch.compile(laplacian_c) if comp else laplacian_c
        self.grad_perp = torch.compile(grad_perp) if comp else grad_perp
        self.interp_TP = torch.compile(interp_TP) if comp else interp_TP
        self.laplacian_h = torch.compile(laplacian_h) if comp else laplacian_h
        if not comp:
            print('Need torch >= 2.0 to use torch.compile, current version '
                 f'{torch.__version__}, the solver will be slower! ')

    def compute_auxillary_matrices(self):
        # function adapted from github.com/louity/MQGeometry
        # Copyright (c) 2023 louity
        nx, ny = self.nx, self.ny
        laplace_dst = compute_laplace_dst(
                nx, ny, self.dx, self.dy, self.arr_kwargs).unsqueeze(0).unsqueeze(0)
        self.helmholtz_dst =  laplace_dst

        laplace_dct = compute_laplace_dct(
                nx, ny, self.dx, self.dy, self.arr_kwargs).unsqueeze(0).unsqueeze(0)
        self.helmholtz_dct =  laplace_dct

        # homogeneous Helmholtz solutions
        cst = torch.ones((1, nx+1, ny+1), **self.arr_kwargs)
        if len(self.masks.psi_irrbound_xids) > 0:
            self.cap_matrices_dst = compute_dst_capacitance_matrices(
                self.helmholtz_dst, self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids)

            sol = solve_helmholtz_dst_cmm(
                    (cst*self.masks.psi)[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices_dst,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            self.cap_matrices_dst = None
            sol = solve_helmholtz_dst(cst[...,1:-1,1:-1], self.helmholtz_dst)

        self.cap_matrices_dct = None

        self.helmholtz_dst = self.helmholtz_dst.type(self.dtype)
        self.helmholtz_dct = self.helmholtz_dct.type(self.dtype)

    def apply_laplacian_c(self,x):
        return self.laplacian_c(x,self.masks.u, self.masks.v, self.n_shift,self.dx,self.dy)*self.masks.q.squeeze()

    def apply_laplacian_c_preconditioner(self,r):
        return (r*self.masks.q).squeeze()

    def laplacian_c_inverse_pcg(self,b):
        """Inverts the laplacian with homogeneous neumann boundary conditions
        defined on the tracer points"""

        x0 = torch.rand(self.xc.shape,**self.arr_kwargs)
        x0 = x0 - x0.mean()

        return pcg(self.apply_laplacian_c, 
                   self.apply_laplacian_c_preconditioner,
                   x0, b,tol=self.pcg_tol, max_iter=self.pcg_max_iter,
                   arr_kwargs = self.arr_kwargs)*self.masks.q.squeeze()

    def laplacian_c_inverse_dct(self,b):
        """Inverts the laplacian with homogeneous neumann boundary conditions
        defined on the tracer points"""
        # function adapted from github.com/louity/MQGeometry
        # Copyright (c) 2023 louity
        
        # if self.cap_matrices is not None:
        #     return solve_helmholtz_dct_cmm(
        #             b[...,1:-1,1:-1]*self.masks.psi[...,1:-1,1:-1],
        #             self.helmholtz_dct, self.cap_matrices,
        #             self.masks.psi_irrbound_xids,
        #             self.masks.psi_irrbound_yids,
        #             self.masks.psi)
        # else:
        return solve_helmholtz_dct(b, self.helmholtz_dct)

    def apply_laplacian_g(self,f):
        fm_g = self.masks.psi.squeeze()*f # Mask the data to apply homogeneous dirichlet boundary conditions
        return self.masks.psi.squeeze()*self.laplacian_g(fm_g,self.masks.psi,self.d_shift,self.dx,self.dy)

    def apply_laplacian_g_preconditioner(self,r):
        """If some dirichlet modes are found already, we use those modes to precondition the system"""
        return (r*self.masks.psi).squeeze()
        

    def laplacian_g_inverse_pcg(self,b):
        """Inverts the laplacian with homogeneous dirichlet boundary conditions
        defined on the vorticity points"""

        x0 = torch.rand(self.nx+1,self.ny+1,**self.arr_kwargs)

        return pcg(self.apply_laplacian_g, 
                   self.apply_laplacian_g_preconditioner,
                   x0, b,tol=self.pcg_tol, max_iter=self.pcg_max_iter,
                   arr_kwargs = self.arr_kwargs)*self.masks.psi.squeeze()

    def laplacian_g_inverse_dst(self,b):
        """Inverts the laplacian with homogeneous dirichlet boundary conditions
        defined on the vorticity points"""
        # function adapted from github.com/louity/MQGeometry
        # Copyright (c) 2023 louity
        
        if self.cap_matrices_dst is not None:
            return solve_helmholtz_dst_cmm(
                    b[...,1:-1,1:-1]*self.masks.psi[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices_dst,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            return solve_helmholtz_dst(b[...,1:-1,1:-1], self.helmholtz_dst)             

    def calculate_dirichlet_modes(self, tol=1e-12, max_iter=100):
        """Uses IRLM to calcualte the N smallest eigenmodes, corresponding to the 
        largest length scales, of the Laplacian operator with homogeneous dirichlet
        boundary conditions."""

        # create an initial guess/seed vector for IRLM
        v0 = self.xg*(self.xg-self.Lx)*self.yg*(self.yg-self.Ly)
        
        evals, eigenvectors, r, last_iterate = implicitly_restarted_lanczos(self.laplacian_g_inverse_pcg, v0,
            self.nmodes, self.nkrylov, tol=tol, max_iter=max_iter,
            arr_kwargs = self.arr_kwargs)

        eigenvalues = 1.0/evals + self.d_shift

        return eigenvalues, eigenvectors, r, last_iterate

    def calculate_neumann_modes(self, tol=1e-12, max_iter=100):
        """Uses IRLM to calcualte the N smallest eigenmodes, corresponding to the 
        largest length scales, of the Laplacian operator with homogeneous dirichlet
        boundary conditions."""

        # create an initial guess/seed vector for IRLM
        v0 = torch.rand(self.xc.shape, **self.arr_kwargs)
        v0 = v0 - v0.mean()
        
        evals, eigenvectors, r, last_iterate = implicitly_restarted_lanczos(self.laplacian_c_inverse_pcg, v0,
            self.nmodes, self.nkrylov, tol=tol, max_iter=max_iter,
            arr_kwargs = self.arr_kwargs)

        eigenvalues = 1.0/evals + self.n_shift

        return eigenvalues, eigenvectors, r, last_iterate
    
  
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

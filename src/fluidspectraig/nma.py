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
                      solve_helmholtz_dst_cmm, compute_capacitance_matrices
from fluidspectraig.masks import Masks
from fluidspectraig.mfeigen_torch import implicitly_restarted_arnoldi,arnoldi_iteration, norm, dot

zeroTol = 1e-12


def laplacian_c(f, dx, dy):
    """2-D laplacian on the tracer points. On tracer points, we are
    working with the divergent modes, which are associated with neumann
    boundary conditions. """
    return F.pad(
        (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1]) / dx**2 \
      + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1]) / dy**2,
        (1,1,1,1), mode='constant', value=0.)

def laplacian_g(f, dx, dy):
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
        (1,1,1,1), mode='constant', value=0.)


class NMA:
    def __init__(self, param):
        self.nx = param['nx']
        self.Lx = param['Lx']
        self.ny = param['ny']
        self.Ly = param['Ly']
        self.nmodes = param['nmodes']
        self.nkrylov = param['nkrylov']
        self.device = param['device']
        self.arr_kwargs = {'dtype':torch.float64, 'device': self.device}

        # grid
        self.xg, self.yg = torch.meshgrid(torch.linspace(0, self.Lx, self.nx+1, **self.arr_kwargs),
                                        torch.linspace(0, self.Ly, self.ny+1, **self.arr_kwargs),
                                        indexing='ij')

        self.dx = torch.tensor(self.Lx / self.nx, **self.arr_kwargs)
        self.dy = torch.tensor(self.Ly / self.ny, **self.arr_kwargs)

        self.xc, self.yc = torch.meshgrid(torch.linspace(self.dx*0.5, self.Lx-self.dx*0.5, self.nx, **self.arr_kwargs),
                                        torch.linspace(self.dy*0.5, self.Ly-self.dy*0.5, self.ny, **self.arr_kwargs),
                                        indexing='ij')

        self.neumann_modes = torch.zeros((self.nmodes,self.nx, self.ny), **self.arr_kwargs) # on tracer points
        self.dirichlet_modes = torch.zeros((self.nmodes,self.nx+1, self.ny+1), **self.arr_kwargs) # on vorticity points

        mask = param['mask'] if 'mask' in param.keys()  else torch.ones(self.nx, self.ny)
        self.masks = Masks(mask.type(torch.float64).to(self.device))

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # precompile torch functions
        comp =  torch.__version__[0] == '2'
        self.laplacian_g = torch.compile(laplacian_g) if comp else laplacian_g
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

        # homogeneous Helmholtz solutions
        cst = torch.ones((1, nx+1, ny+1), **self.arr_kwargs)
        if len(self.masks.psi_irrbound_xids) > 0:
            self.cap_matrices = compute_capacitance_matrices(
                self.helmholtz_dst, self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids)
            sol = solve_helmholtz_dst_cmm(
                    (cst*self.masks.psi)[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            self.cap_matrices = None
            sol = solve_helmholtz_dst(cst[...,1:-1,1:-1], self.helmholtz_dst)

        self.helmholtz_dst = self.helmholtz_dst.type(torch.float32)

    def laplacian_g_inverse(self,b):
        """Inverts the laplacian with homogeneous dirichlet boundary conditions
        defined on the vorticity points"""
        # function adapted from github.com/louity/MQGeometry
        # Copyright (c) 2023 louity
        
        if self.cap_matrices is not None:
            return solve_helmholtz_dst_cmm(
                    b[...,1:-1,1:-1]*self.masks.psi[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            return solve_helmholtz_dst(b[...,1:-1,1:-1], self.helmholtz_dst)

    def apply_laplacian_g(self,f):
        fm_g = self.masks.psi*f
        return self.masks.psi*self.laplacian_g(f,self.dx,self.dy)
        
    def calculate_dirichlet_modes(self, tol=1e-6, max_iter=100):
        """Uses IRAM to calcualte the N smallest eigenmodes,
        corresponding to the largest length scales, of the operator
        (L - sI), where L is the laplacian with homogeneous dirichlet
        boundary conditions, s is a scalar shift, and I is the 
        identity matrix """

        # need an initial guess generator that 
        # creates a function that contains data similar
        # to the modes we're looking for.
        v0 = torch.sin(self.xg*torch.pi/self.Lx)*torch.sin(self.yg*torch.pi/self.Ly) + torch.rand(self.xg.shape,**self.arr_kwargs) # generate seed vector on vorticity points
        
        eigenvalues, eigenvectors, r = implicitly_restarted_arnoldi(self.laplacian_g_inverse, v0,
            self.nmodes, self.nkrylov, tol=tol, max_iter=max_iter, 
            arr_kwargs = self.arr_kwargs)

        return eigenvalues, eigenvectors, r

if __name__ == '__main__':

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    param = {'nx': 100,
             'ny': 100,
             'Lx': 1.0,
             'Ly': 1.0,
             'nmodes': 80,
             'device': 'cuda'}
   
    model = NMA(param)
    
    
    # # def save(self, filename):
    # #     """Saves the model and eigenpairs to npz file"""

    # #     import numpy as np

    # #     np.savez(filename,xc=self.xc,xg=self.xg,
    # #         yc=self.yc,yg=self.yg,maskC=self.maskC,
    # #         maskZ=self.maskZ,maskS=self.maskS,
    # #         maskW=self.maskW,hfacC=self.hfacC,
    # #         d_eigenvalues=self.d_eigenvalues,
    # #         d_eigenmodes=self.d_eigenmodes,
    # #         n_eigenvalues=self.n_eigenvalues,
    # #         n_eigenmodes=self.n_eigenmodes)

    
    # def LapZInv_PCCG(
    #     self, b, s0=None, pcitermax=20, pctolerance=1e-2, itermax=1500, tolerance=1e-4, shift=0.0
    # ):
    #     """Uses preconditioned conjugate gradient to solve L s = b,
    #     where `L s` is the Laplacian on vorticity points applied to s
    #     Stopping criteria is when the relative change in the solution is
    #     less than the tolerance.

    #     The preconditioner is the LapZInv_JacobianSolve method.

    #     Algorithm taken from pg.51 of
    #     https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    #     """
    #     import numpy as np
    #     import xnma.kernels as kernels

    #     if s0:
    #         sk = s0
    #     else:
    #         sk = np.zeros_like(b)

    #     sk = sk * self.maskZ

    #     r = kernels.LapZ_Residual(
    #         sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
    #     )

    #     d = self.LapZInv_JacobiSolve(r, itermax=pcitermax, tolerance=pctolerance, shift=shift)

    #     delta = np.sum(r * d)
    #     rmag = np.sqrt(np.sum(r*r))
    #     bmag = np.sqrt(np.sum(b*b))
    #     r0 = np.max([rmag, bmag])

    #     for k in range(0, itermax):
    #         q = (
    #             kernels.LapZ(d, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift)
    #             * self.maskZ
    #         )

    #         alpha = delta / (np.sum(d * q))

    #         sk += alpha * d
    #         if k % 50 == 0:
    #             r = kernels.LapZ_Residual(
    #                 sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
    #             )
    #         else:
    #             r -= alpha * q

    #         x = self.LapZInv_JacobiSolve(r, itermax=pcitermax, tolerance=pctolerance, shift=shift)

    #         rmag = np.sqrt(np.sum(r*r))
    #         deltaOld = delta
    #         delta = np.sum(r * x)
    #         beta = delta / deltaOld
    #         d = x + beta * d
    #         if rmag <= tolerance*r0:
    #             break

    #     if rmag > tolerance*r0:
    #         print(
    #             f"Conjugate gradient method did not converge in {k+1} iterations : {delta}"
    #         )

    #     return sk

    # def LapZInv_JacobiSolve(self, b, s0=None, itermax=1000, tolerance=1e-4, shift=0.0):
    #     """Performs Jacobi iterations to iteratively solve L s = b,
    #     where `L s` is the Laplacian on vorticity points with homogeneous Dirichlet
    #     boundary conditions applied to s
    #     Stopping criteria is when the relative change in the solution is
    #     less than the tolerance.

    #     !!! warning
    #         This tolerance is invalid when max(abs(b)) == max(abs(s)) = 0
    #     """
    #     import numpy as np
    #     import xnma.kernels as kernels

    #     if s0:
    #         sk = s0
    #     else:
    #         sk = np.zeros_like(b)

    #     sk = sk * self.maskZ
    #     r = kernels.LapZ_Residual(
    #         sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
    #     )

    #     for k in range(0, itermax):
    #         ds = kernels.LapZ_JacobiDinv(
    #             r, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
    #         )

    #         dsmag = np.max(abs(ds))
    #         smag = np.max(abs(sk))
    #         if smag <= np.finfo(self.prec).eps:
    #             smag == np.max(abs(b))

    #         if smag > np.finfo(self.prec).eps:
    #             if dsmag / smag <= tolerance:
    #                 break

    #         # Update the solution
    #         sk += ds

    #         r = kernels.LapZ_Residual(
    #             sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
    #         )

    #     return sk

    # def LapCInv_PCCG(
    #     self,
    #     b,
    #     s0=None,
    #     pcitermax=0,
    #     pctolerance=1e-2,
    #     itermax=1500,
    #     tolerance=1e-4,
    #     dShift=1e-2,
    # ):
    #     """Uses preconditioned conjugate gradient to solve L s = b,
    #     where `L s` is the Laplacian on tracer points applied to s
    #     Stopping criteria is when the relative change in the solution is
    #     less than the tolerance.

    #     The preconditioner is the LapCInv_JacobianSolve method.

    #     Algorithm taken from pg.51 of
    #     https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    #     """
    #     import numpy as np
    #     import xnma.kernels as kernels

    #     # if pcitermax > 0:
    #     #     print("Warning : Jacobi preconditioner unverified for homogeneous neumann modes")

    #     if s0:
    #         sk = s0
    #     else:
    #         sk = np.zeros_like(b)

    #     sk = sk * self.maskC

    #     r = kernels.LapC_Residual(
    #         sk,
    #         b,
    #         self.maskC,
    #         self.dxc,
    #         self.dyc,
    #         self.dxg,
    #         self.dyg,
    #         self.maskW,
    #         self.maskS,
    #         self.rac,
    #         dShift,
    #     )

    #     if pcitermax == 0:
    #         d = r * self.maskC
    #     else:
    #         d = self.LapCInv_JacobiSolve(
    #             r, itermax=pcitermax, tolerance=pctolerance, dShift=dShift
    #         )

    #     delta = np.sum(r * d)
    #     rmag = np.sqrt(np.sum(r*r))
    #     bmag = np.sqrt(np.sum(b*b))
    #     r0 = np.max([rmag, bmag])

    #     for k in range(0, itermax):
    #         # print(f"(k,r) : ({k},{rmag})")
    #         q = (
    #             kernels.LapC(
    #                 d,
    #                 self.dxc,
    #                 self.dyc,
    #                 self.dxg,
    #                 self.dyg,
    #                 self.maskW,
    #                 self.maskS,
    #                 self.rac,
    #                 dShift,
    #             )
    #             * self.maskC
    #         )

    #         alpha = delta / (np.sum(d * q))

    #         sk += alpha * d
    #         if k % 50 == 0:
    #             r = kernels.LapC_Residual(
    #                 sk,
    #                 b,
    #                 self.maskC,
    #                 self.dxc,
    #                 self.dyc,
    #                 self.dxg,
    #                 self.dyg,
    #                 self.maskW,
    #                 self.maskS,
    #                 self.rac,
    #                 dShift,
    #             )
    #         else:
    #             r -= alpha * q

    #         if pcitermax == 0:
    #             x = r * self.maskC
    #         else:
    #             x = self.LapCInv_JacobiSolve(
    #                 r, itermax=pcitermax, tolerance=pctolerance, dShift=dShift
    #             )

    #         rmag = np.sqrt(np.sum(r*r))
    #         deltaOld = delta
    #         delta = np.sum(r * x)
    #         beta = delta / deltaOld
    #         d = x + beta * d
    #         if rmag <= tolerance*r0 :
    #             break

    #     if rmag > tolerance*r0:
    #         print(
    #             f"Conjugate gradient method did not converge in {k+1} iterations : {rmag}"
    #         )

    #     return sk

    # def LapCInv_JacobiSolve(
    #     self, b, s0=None, itermax=1000, tolerance=1e-4, dShift=1e-2
    # ):
    #     """Performs Jacobi iterations to iteratively solve L s = b,
    #     where `L s` is the Laplacian on tracer points applied to s
    #     Stopping criteria is when the relative change in the solution is
    #     less than the tolerance.

    #     !!! warning
    #         This tolerance is invalid when max(abs(b)) == max(abs(s)) = 0
    #     """
    #     import numpy as np
    #     import xnma.kernels as kernels

    #     if s0:
    #         sk = s0
    #     else:
    #         sk = np.zeros_like(b)

    #     sk = sk * self.maskC
    #     r = kernels.LapC_Residual(
    #         sk,
    #         b,
    #         self.maskC,
    #         self.dxc,
    #         self.dyc,
    #         self.dxg,
    #         self.dyg,
    #         self.maskW,
    #         self.maskS,
    #         self.rac,
    #         dShift,
    #     )

    #     for k in range(0, itermax):
    #         ds = kernels.LapC_JacobiDinv(
    #             r,
    #             self.dxc,
    #             self.dyc,
    #             self.dxg,
    #             self.dyg,
    #             self.maskW,
    #             self.maskS,
    #             self.rac,
    #             dShift,
    #         )

    #         dsmag = np.max(abs(ds))
    #         smag = np.max(abs(sk))
    #         if smag <= np.finfo(self.prec).eps:
    #             smag == np.max(abs(b))

    #         if smag > np.finfo(self.prec).eps:
    #             if dsmag / smag <= tolerance:
    #                 break

    #         # Update the solution
    #         sk += ds

    #         r = kernels.LapC_Residual(
    #             sk,
    #             b,
    #             self.maskC,
    #             self.dxc,
    #             self.dyc,
    #             self.dxg,
    #             self.dyg,
    #             self.maskW,
    #             self.maskS,
    #             self.rac,
    #             dShift,
    #         )

    #     return sk

    # def laplacianZ(self, x, dShift):
    #     """Wrapper for the Laplacian, where x comes in as a flat 1-D array
    #     only at `wet` grid cell locations"""

    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels

    #     # x comes in as a 1-D array in "DOF" format
    #     # we need to convert it to a 2-D array consistent with the model grid
    #     xgrid = ma.masked_array(
    #         np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=self.prec
    #     )
    #     xgrid[~xgrid.mask] = x  # Set interior values to b

    #     # Invert the laplacian
    #     Lx = kernels.LapZ(x, self.dxc, self.dyc, self.dxg, self.dyg, self.raz)

    #     # Mask the data, so that we can return a 1-D array of unmasked values
    #     return ma.masked_array(
    #         Lx, mask=abs(self.maskZ - 1.0), dtype=self.prec
    #     ).compressed()

    # def laplacianZInverse(self, b, dShift):
    #     """Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
    #     where b comes in as a flat 1-D array only at `wet` grid cell locations"""

    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels

    #     # import time

    #     # b comes in as a 1-D array in "DOF" format
    #     # we need to convert it to a 2-D array consistent with the model grid
    #     # Use the model.b attribute to push the DOF formatted data

    #     bgrid = ma.masked_array(
    #         np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=self.prec
    #     )
    #     bgrid[~bgrid.mask] = b  # Set interior values to b

    #     x = np.ones(self.maskZ.shape, dtype=self.prec)
    #     # Invert the laplacian
    #     x = self.LapZInv_PCCG(
    #         bgrid.data,
    #         s0=None,
    #         pcitermax=20,
    #         pctolerance=1e-2,
    #         itermax=3000,
    #         tolerance=1e-14,
    #         shift=dShift
    #     )

    #     # Mask the data, so that we can return a 1-D array of unmasked values
    #     return ma.masked_array(
    #         x, mask=abs(self.maskZ - 1.0), dtype=self.prec
    #     ).compressed()

    # def laplacianC(self, x, dShift):
    #     """Wrapper for the Laplacian, where x comes in as a flat 1-D array
    #     only at `wet` grid cell locations"""

    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels

    #     # x comes in as a 1-D array in "DOF" format
    #     # we need to convert it to a 2-D array consistent with the model grid
    #     xgrid = ma.masked_array(
    #         np.zeros(self.maskC.shape), mask=abs(self.maskC - 1.0), dtype=self.prec
    #     )
    #     xgrid[~xgrid.mask] = x  # Set interior values to b

    #     # Invert the laplacian
    #     Lx = kernels.LapC(
    #         x,
    #         self.dxc,
    #         self.dyc,
    #         self.dxg,
    #         self.dyg,
    #         self.maskW,
    #         self.maskS,
    #         self.raC,
    #         dShift,
    #     )

    #     # Mask the data, so that we can return a 1-D array of unmasked values
    #     return ma.masked_array(
    #         Lx, mask=abs(self.maskC - 1.0), dtype=self.prec
    #     ).compressed()

    # def laplacianCInverse(self, b, dShift):
    #     """Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
    #     where b comes in as a flat 1-D array only at `wet` grid cell locations"""

    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels
    #     import time

    #     # b comes in as a 1-D array in "DOF" format
    #     # we need to convert it to a 2-D array consistent with the model grid
    #     # Use the model.b attribute to push the DOF formatted data

    #     bgrid = ma.masked_array(
    #         np.zeros(self.maskC.shape), mask=abs(self.maskC - 1.0), dtype=self.prec
    #     )
    #     bgrid[~bgrid.mask] = b  # Set interior values to b

    #     x = np.ones(self.maskC.shape, dtype=self.prec)
    #     # Invert the laplacian
    #     x = self.LapCInv_PCCG(
    #         bgrid.data,
    #         s0=None,
    #         pcitermax=20,
    #         pctolerance=1e-2,
    #         itermax=3000,
    #         tolerance=1e-14,
    #         dShift=dShift,
    #     )

    #     # Mask the data, so that we can return a 1-D array of unmasked values
    #     return ma.masked_array(
    #         x, mask=abs(self.maskC - 1.0), dtype=self.prec
    #     ).compressed()

    # def findDirichletModes(self, nmodes=10, tolerance=0, shift=0):
    #     """Find the eigenpairs associated with the Laplacian operator on
    #     vorticity points with homogeneous Dirichlet boundary conditions.

    #     """
    #     from scipy.sparse.linalg import LinearOperator
    #     from scipy.sparse.linalg import eigsh
    #     import time
    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels

    #     shape = (self.ndofZ, self.ndofZ)
    #     Linv = LinearOperator(
    #         shape, matvec=lambda b: self.laplacianZInverse(b, shift), dtype=self.prec
    #     )

    #     print("[Dirichlet modes] Starting eigenvalue search")
    #     tic = time.perf_counter()
    #     evals, evecs = eigsh(Linv, k=nmodes, tol=tolerance, return_eigenvectors=True)
    #     toc = time.perf_counter()
    #     runtime = toc - tic
    #     print(f"[Dirichlet modes] eigsh runtime : {runtime:0.4f} s")

    #     sgrid = ma.masked_array(
    #         np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=self.prec
    #     )
    #     ny, nx = self.maskZ.shape
    #     eigenvalues = np.zeros((nmodes), dtype=self.prec)
    #     eigenmodes = np.zeros((nmodes, ny, nx), dtype=self.prec)

    #     for k in range(0, nmodes):
    #         ev = 1.0 / evals[k] + shift
    #         if np.abs(ev) < np.abs(zeroTol):
    #             eigenvalues[k] = 0.0
    #         else:
    #             eigenvalues[k] = ev

    #         # Interpolate the dirichlet modes from the vorticity points
    #         # to the tracer points and store the result in sgrid
    #         sgrid[~sgrid.mask] = evecs[:, k]
    #         g = sgrid.data * self.maskZ

    #         # Normalize so that the norm of the eigenmode is 1
    #         mag = np.sqrt(np.sum(g * g * self.raz))
    #         eigenmodes[k, :, :] = g / mag

    #     self.d_eigenvalues = eigenvalues
    #     self.d_eigenmodes = eigenmodes

    # def findNeumannModes(self, nmodes=10, tolerance=0, shift=1e-2):
    #     """Find the eigenpairs associated with the Laplacian operator on
    #     tracer points with homogeneous Neumann boundary conditions.

    #     """
    #     from scipy.sparse.linalg import LinearOperator
    #     from scipy.sparse.linalg import eigsh
    #     import time
    #     import numpy as np
    #     from numpy import ma
    #     import xnma.kernels as kernels

    #     shape = (self.ndofC, self.ndofC)
    #     Linv = LinearOperator(
    #         shape, matvec=lambda b: self.laplacianCInverse(b, shift), dtype=self.prec
    #     )
    #     print("[Neumann modes] Starting eigenvalue search")
    #     tic = time.perf_counter()
    #     evals, evecs = eigsh(Linv, k=nmodes, tol=tolerance, return_eigenvectors=True)
    #     toc = time.perf_counter()
    #     runtime = toc - tic
    #     print(f"[Neumann modes] eigsh runtime : {runtime:0.4f} s")

    #     ny, nx = self.maskC.shape
    #     eigenvalues = np.zeros((nmodes), dtype=self.prec)
    #     eigenmodes = np.zeros((nmodes, ny, nx), dtype=self.prec)
    #     sgrid = ma.masked_array(
    #         np.zeros(self.maskC.shape), mask=abs(self.maskC - 1.0), dtype=self.prec
    #     )
        
    #     for k in range(0, nmodes):
    #         ev = 1.0 / evals[k] + shift
    #         if np.abs(ev) < np.abs(zeroTol):
    #             eigenvalues[k] = 0.0
    #         else:
    #             eigenvalues[k] = ev

    #         sgrid[~sgrid.mask] = evecs[:, k]
    #         g = sgrid.data * self.maskC

    #         # Normalize so that the norm of the eigenmode is 1
    #         mag = np.sqrt(np.sum(g * g * self.rac))
    #         eigenmodes[k, :, :] = g / mag

    #     self.n_eigenvalues = eigenvalues
    #     self.n_eigenmodes = eigenmodes

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

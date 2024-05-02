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

"""
Defines matrix-free and elliptic operators and preconditioners for use with nma_torch
"""
import torch.nn.functional as F

def dfdx_c(f,dx):
    """Calculates the x-derivative of a function on tracer points
    and returns a function on u-points.Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,1:,:]-f[...,:-1,:])/dx, (0,0,1,1), mode='constant',value=0.
    )

def dfdx_v(f,dx):
    """Calculates the x-derivative of a function on v points
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

def dfdy_u(f,dy):
    """Calculates the y-derivative of a function on tracer points
    and returns a function on u-points. Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,:,1:]-f[...,:,:-1])/dy, (1,1,0,0), mode='constant',value=0.
    )

def dfdy_v(f,dy):
    """Calculates the y-derivative of a function on v points
    and returns a function on tracer points."""
    return (f[...,:,1:]-f[...,:,:-1])/dy

def laplacian_c(f, masku, maskv, dx, dy):
    """2-D laplacian on the tracer points. On tracer points, we are
    working with the divergent modes, which are associated with neumann
    boundary conditions. """
    return dfdx_u( dfdx_c(f,dx)*masku, dx ) + dfdy_v( dfdy_c(f,dy)*maskv, dy )


def laplacian_g(f, maskz, dx, dy):
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
        (1,1,1,1), mode='constant', value=0.)*maskz


def TtoU(f):
    """Interpolates from arakawa c-grid tracer point to u-point.
    Input is first padded in the x-direction to prolong the data
    past the boundaries, consistent with homogeneous neumann conditions
    for data at tracer points. """
    fpad = F.pad( f[...,:,:], (0,0,1,1), mode='replicate')
    return 0.5*(fpad[...,1:,:]+fpad[...,:-1,:])

def TtoV(f):
    """Interpolates from arakawa c-grid tracer point to v-point.
    Input is first padded in the x-direction to prolong the data
    past the boundaries, consistent with homogeneous neumann conditions
    for data at tracer points. """
    fpad = F.pad( f, (1,1,0,0), mode='replicate')
    return 0.5*(fpad[...,:,1:]+fpad[...,:,:-1])       

def vorticity_cgrid(u,v,maskz,dx,dy):
    return (dfdx_v(v,dx) - dfdy_u(u,dy))*maskz

def divergence_cgrid(u,v,maskc,dx,dy):
    return (dfdx_u(u,dx) + dfdy_v(v,dy))*maskc

def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[...,:-1] - f[...,1:]) / dy, (f[...,1:,:] - f[...,:-1,:]) / dx


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
        self.area_n = self.masks.q*self.dx*self.dy # area of neumann mode cells
        self.area_d = self.masks.psi*self.dx*self.dy # area of dirichlet mode cells

        # precompile torch functions
        comp =  torch.__version__[0] == '2'
        self.laplacian_g = torch.compile(laplacian_g) if comp else laplacian_g
        self.laplacian_c = torch.compile(laplacian_c) if comp else laplacian_c

        if not comp:
            print('Need torch >= 2.0 to use torch.compile, current version '
                 f'{torch.__version__}, the solver will be slower! ')


    def apply_laplacian_n(self,x):
        """Laplacian with neumann boundary conditions"""
        return -self.laplacian_c(x,self.masks.u,self.masks.v,self.dx,self.dy)*self.masks.q.squeeze()


    def apply_laplacian_d(self,f):
        """Laplacian with dirichlet boundary conditions"""
        fm_g = self.masks.psi.squeeze()*f # Mask the data to apply homogeneous dirichlet boundary conditions
        return -self.masks.psi.squeeze()*self.laplacian_g(fm_g,self.masks.psi,self.dx,self.dy)

    def vorticity(self,u,v):
        return vorticity_cgrid(u,v,self.masks.psi,self.dx,self.dy)

    def divergence(self,u,v):
        return divergence_cgrid(u,v,self.masks.q,self.dx,self.dy)

    def map_T_to_U(self,f):
        return TtoU(f)

    def map_T_to_V(self,f):
        return TtoV(f)

    def total_area_d(self):
        return torch.sum( self.area_d ) 
    
    def total_area_n(self):
        return torch.sum( self.area_n ) 



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
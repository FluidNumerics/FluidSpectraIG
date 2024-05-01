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

import torch.nn.functional as F

def dfdx_c(f,dxc):
    """Calculates the x-derivative of a function on tracer points
    and returns a function on u-points.Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,1:,:]-f[...,:-1,:])/dxc[:,:], (0,0,1,1), mode='constant',value=0.
    )

def dfdx_v(f,dxc):
    """Calculates the x-derivative of a function on v points
    and returns a function on u-points.Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,1:,:]-f[...,:-1,:])/dxc[:,:], (0,0,1,1), mode='constant',value=0.
    )
    
def dfdx_u(f,dxc):
    """Calculates the x-derivative of a function on u points
    and returns a function on tracer-points."""
    return (f[...,1:,:]-f[...,:-1,:])/dxc[:,:]

def dfdy_c(f,dyc):
    """Calculates the y-derivative of a function on tracer points
    and returns a function on v-points. Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,:,1:]-f[...,:,:-1])/dyc[:,:], (1,1,0,0), mode='constant',value=0.
    )

def dfdy_u(f,dyc):
    """Calculates the y-derivative of a function on tracer points
    and returns a function on u-points. Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,:,1:]-f[...,:,:-1])/dyc[:,:], (1,1,0,0), mode='constant',value=0.
    )

def dfdy_v(f,dyg):
    """Calculates the y-derivative of a function on v points
    and returns a function on tracer points."""
    return (f[...,:,1:]-f[...,:,:-1])/dyg[:,:]

def delta_ir(f):
    """Returns right sided difference in the i-dimension"""
    return (f[...,1:,:]-f[...,:-1,:])

def delta_jr(f):
    """Returns top sided difference in the j-dimension"""
    return (f[...,:,1:]-f[...,:,:-1])

def laplacian_c(f, masku, maskv, dxc, dyc, dxg, dyg, rac):
    """2-D laplacian on the tracer points. On tracer points, we are
    working with the divergent modes, which are associated with neumann
    boundary conditions. """

    gx = F.pad(delta_ir(f)/dxc,(0,0,1,1),mode='constant',value=0.0)*dyg
    gy = F.pad(delta_jr(f)/dyc,(1,1,0,0),mode='constant',value=0.0)*dxg
    return (delta_ir(gx) + delta_jr(gy))/rac

def laplacian_g(f, maskz, dxc, dyc, dxg, dyg, raz):
    """2-D laplacian on the vorticity points. On vorticity points, we are
    working with the rotational modes, which are associated with dirichlet 
    boundary conditions. Function values are assumed to be masked prior
    to calling this method. Additionally, the laplacian is returned
    as zero numerical boundaries"""

    gx = (delta_ir(f)[...,:,1:-1])*dyc/dxg[:,1:-1]
    gy = (delta_jr(f)[...,1:-1,:])*dxc/dyg[1:-1,:]
    del2f = (delta_ir(gx) + delta_jr(gy))/raz
    return F.pad( del2f, (1,1,1,1), mode='constant', value=0.0 )*maskz

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

def divergence_cgrid(u,v,maskc,dxg,dyg):
    return (dfdx_u(u,dxg) + dfdy_v(v,dyg))*maskc



class MITgcm:
    """Data structure for working with the MITgcm"""
    def __init__(self, param):

        self.device = param['device']
        self.dtype = torch.float64            
        self.arr_kwargs = {'dtype':self.dtype, 'device': self.device}

        if 'case_directory' in param.keys():
            self.load(param['case_directory'])
        else:
            self.nx = param['nx']
            self.Lx = param['Lx']
            self.ny = param['ny']
            self.Ly = param['Ly']
            self.dx = self.Lx / self.nx 
            self.dy = self.Ly / self.ny 

            # grid
            # > vorticity points
            self.xg, self.yg = torch.meshgrid(torch.linspace(0, self.Lx, self.nx+1, **self.arr_kwargs),
                                        torch.linspace(0, self.Ly, self.ny+1, **self.arr_kwargs),
                                        indexing='ij') # size(nx+1,ny+1)
            # > tracer points
            self.xc, self.yc = torch.meshgrid(torch.linspace(self.dx*0.5, self.Lx-self.dx*0.5, self.nx, **self.arr_kwargs),
                                        torch.linspace(self.dy*0.5, self.Ly-self.dy*0.5, self.ny, **self.arr_kwargs),
                                        indexing='ij') # size(nx,ny)
                          
            self.dxg = delta_ir(self.xg) # size(nx,ny+1)
            self.dyg = delta_jr(self.yg) # size(nx+1,ny)

            self.dxc = delta_ir(self.xc) # size(nx-1,ny)
            self.dyc = delta_jr(self.yc) # size(nx,ny-1)

            # RAZ is the area of vorticity cells at interior points; this excludes rectangular domain boundary points (size: [nx-2 ,ny-2])
            self.raz = self.dxc[:,:-1]*self.dyc[:-1,:]
            # RAC is the area of tracer cells at interior points; size(nx,ny)
            self.rac = self.dxg[:,:-1]*self.dyg[:-1,:]

            print(f"shape(xg) : {self.xg.shape}")
            print(f"shape(yg) : {self.xg.shape}")
            print(f"shape(xc) : {self.xc.shape}")
            print(f"shape(yc) : {self.xc.shape}")
            print(f"shape(dxg) : {self.dxg.shape}")
            print(f"shape(dyg) : {self.dyg.shape}")
            print(f"shape(dxc) : {self.dxc.shape}")
            print(f"shape(dyc) : {self.dyc.shape}")
            print(f"shape(raz) : {self.raz.shape}")
            print(f"shape(rac) : {self.rac.shape}")

            mask = param['mask'] if 'mask' in param.keys()  else torch.ones(self.nx, self.ny)
            self.masks = Masks(mask.type(self.dtype).to(self.device))

        # precompile torch functions
        comp =  torch.__version__[0] == '2'
        self.laplacian_g = torch.compile(laplacian_g) if comp else laplacian_g
        self.laplacian_c = torch.compile(laplacian_c) if comp else laplacian_c

        if not comp:
            print('Need torch >= 2.0 to use torch.compile, current version '
                 f'{torch.__version__}, the solver will be slower! ')

    def load(self, case_directory, x=None, y=None, depth=0, geometry="sphericalpolar"):
        """Loads in grid from MITgcm metadata files in dataDir
           and configures masks at given depth"""
        import numpy as np
        import xmitgcm
        import xgcm
        from dask import array as da

        chunks = None
        self.depth = depth
        self.case_directory = case_directory

        ds = xmitgcm.open_mdsdataset(
            dataDir,
            iters=None,
            prefix=[],
            read_grid=True,
            geometry=geometry,
        )

        if x:
            ds = ds.sel(XC=slice(x[0], x[1]), XG=slice(x[0], x[1]))
        if y:
            ds = ds.sel(YC=slice(y[0], y[1]), YG=slice(y[0], y[1]))

        # Point locations
        self.xg = torch.from_numpy(ds.XG.to_numpy(),**self.arr_kwargs)
        self.yg = torch.from_numpy(ds.YG.to_numpy(),**self.arr_kwargs)
        
        # Grid spacing
        self.dxg = torch.from_numpy(ds.dxG.to_numpy(),**self.arr_kwargs)
        self.dyg = torch.from_numpy(ds.dyG.to_numpy(),**self.arr_kwargs)

        self.dxc = torch.from_numpy(ds.dxC.to_numpy(),**self.arr_kwargs)
        self.dyc = torch.from_numpy(ds.dyC.to_numpy(),**self.arr_kwargs)

        # Cell areas
        self.rac = torch.from_numpy(ds.rA.to_numpy(),**self.arr_kwargs)
        self.raz = torch.from_numpy(ds.rAz.to_numpy(),**self.arr_kwargs)

        # Get the masks
        self.zd = -abs(depth)  # ensure that depth is negative value
        mask = torch.from_numpy(np.ceil(ds.hFacC.interp(Z=[zd], method="nearest").squeeze().to_numpy()))
        self.nx, self.ny = mask.shape
        self.masks = Masks(mask.type(self.dtype).to(self.device))

        



    #     self.maskC = np.ceil(self.hFacC)
    #     self.maskW = np.ceil(self.hFacW)
    #     self.maskS = np.ceil(self.hFacS)


    def apply_laplacian_c(self,x):
        return -self.laplacian_c(x,self.masks.u,self.masks.v,self.dxc,self.dyc,self.dxg,self.dyg,self.rac)*self.masks.q.squeeze()

    def apply_laplacian_g(self,f):
        fm_g = self.masks.psi.squeeze()*f # Mask the data to apply homogeneous dirichlet boundary conditions
        return -self.masks.psi.squeeze()*self.laplacian_g(fm_g,self.masks.psi,self.dxc,self.dyc,self.dxg,self.dyg,self.raz)

    def model_area_g(self):
        return torch.sum( self.masks.psi*self.raz) 
    
    def model_area_c(self):
        return torch.sum( self.masks.q*self.rac ) 


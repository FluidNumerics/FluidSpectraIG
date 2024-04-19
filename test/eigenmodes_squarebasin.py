#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
from fluidspectraig.mfeigen_torch import dot
import time
from torch.profiler import profile, record_function, ProfilerActivity

def DirichletModes( model ):
    """ Calculates the exact dirichlet modes for a rectangular domain"""

    nx = int((model.nx))
    ny = int((model.ny))
    Lx = model.Lx
    Ly = model.Ly
    nmodes = nx*ny
    #eigenmodes = np.zeros((nmodes,nx+1,ny+1))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nx)
    for m in range(0, nx):
      kx[m] = (m+1)*np.pi/Lx
        
    ky = np.zeros(ny)
    for m in range(0, ny):
      ky[m] = (m+1)*np.pi/Ly

    k = 0
    #tmp = np.zeros((nmodes,nx+1,ny+1))
    ev = np.zeros(nmodes)
    xg = model.xg.cpu().numpy()
    yg = model.yg.cpu().numpy()
    for m in range(0, ny):
        for n in range(0, nx):
            x = xg*kx[n]
            y = yg*ky[m]
            #tmp[k,:,:] = np.sin( x )*np.sin( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]
    #for k in range(0,nmodes):
    #    eigenmodes[k,:,:] = tmp[sort_index[k],:,:].squeeze()

    return eigenvalues#, eigenmodes

def NeumannModes( model ):
    """ Calculates the exact neumann modes for a rectangular domain"""

    nx = int((model.nx))
    ny = int((model.ny))
    Lx = model.Lx
    Ly = model.Ly
    nmodes = nx*ny
    #eigenmodes = np.zeros((nmodes,nx+1,ny+1))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nx)
    for m in range(0, nx):
      kx[m] = (m)*np.pi/Lx
        
    ky = np.zeros(ny)
    for m in range(0, ny):
      ky[m] = (m)*np.pi/Ly

    k = 0
    #tmp = np.zeros((nmodes,nx+1,ny+1))
    ev = np.zeros(nmodes)
    xg = model.xg.cpu().numpy()
    yg = model.yg.cpu().numpy()
    for m in range(0, ny):
        for n in range(0, nx):
            x = xg*kx[n]
            y = yg*ky[m]
          #  tmp[k,:,:] = np.cos( x )*np.cos( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]
    #for k in range(0,nmodes):
    #    eigenmodes[k,:,:] = tmp[sort_index[k],:,:].squeeze()

    return eigenvalues #, eigenmodes

nx = 64
ny = 64
Lx = 1.0#5120.0e3
Ly = 1.0#5120.0e3
nmodes = 256
nclusters = 16
nkrylov = 260

# nx = 256
# ny = 256
# Lx = 1.0#5120.0e3
# Ly = 1.0#5120.0e3
# nmodes = 100
# nkrylov = 115

param = {'nx': nx,
         'ny': ny,
         'Lx': 1.0,
         'Ly': 1.0,
         'krylov_tol':1e-21,
         'preconditioner':'jacobi',
         'device': 'cuda',
         'dtype': torch.float64}

model = NMA(param)


n_eigenvalues, n_eigenvectors, d_eigenvalues, d_eigenvectors = model.eigenmode_search(nclusters=nclusters,nmodes=nmodes,nkrylov=nkrylov,tol=1e-9,max_iter=200)


hf = h5py.File(f'nma_torch_eigenmodes_squarebasin_{nx}-{ny}.h5', 'w')
hf.create_dataset('nkrylov', data=nkrylov)
hf.create_dataset('nmodes', data=nmodes)
hf.create_dataset('nclusters', data=nclusters)

g1 = hf.create_group('neumann')
g1.create_dataset('eigenvalues', data=n_eigenvalues.cpu().numpy())
g1.create_dataset('eigenvectors', data=n_eigenvectors.cpu().numpy())

g2 = hf.create_group('dirichlet')
g2.create_dataset('eigenvalues', data=d_eigenvalues.cpu().numpy())
g2.create_dataset('eigenvectors', data=d_eigenvectors.cpu().numpy())

g3 = hf.create_group('grid')
g3.create_dataset('xc', data=model.xc.cpu().numpy())
g3.create_dataset('yc', data=model.yc.cpu().numpy())
g3.create_dataset('xg', data=model.xg.cpu().numpy())
g3.create_dataset('yg', data=model.yg.cpu().numpy())
g3.create_dataset('mask_q', data=model.masks.q.cpu().numpy())
g3.create_dataset('mask_z', data=model.masks.psi.cpu().numpy())
g3.create_dataset('mask_u', data=model.masks.u.cpu().numpy())
g3.create_dataset('mask_v', data=model.masks.v.cpu().numpy())
hf.close()

nmodes = nmodes*nclusters

# verify orthogonality
PtP = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
PtP_n = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
for row in range(nmodes):
  for col in range(nmodes):
    PtP[row,col] = torch.sum( d_eigenvectors[:,:,row]*d_eigenvectors[:,:,col] )
    PtP_n[row,col] = torch.sum( n_eigenvectors[:,:,row]*n_eigenvectors[:,:,col] )


plt.figure(figsize=(1,1))
A = PtP.cpu().numpy()
plt.imshow(A,interpolation='nearest', aspect='equal')
plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.savefig('orthogonality-verification_d.png',dpi=A.shape[0])
plt.close()

plt.figure(figsize=(1,1))
A = PtP_n.cpu().numpy()
plt.imshow(A,interpolation='nearest', aspect='equal')
plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.savefig('orthogonality-verification_n.png',dpi=A.shape[0])
plt.close()

# # d_eigenvalues = DirichletModes(model)
# # n_eigenvalues = NeumannModes(model)

# f,a = plt.subplots(1,2)
# me = min(d_eigenvalues[0:nmodes])
# Me = max(d_eigenvalues[0:nmodes])
# a[0].set_title(f"Dirichlet Eigenvalues {model.nx} x {model.ny}")
# a[0].plot(np.abs(d_eigenvalues[0:nmodes]), np.abs(eigenvalues.cpu().numpy()),'o',label = 'dirichlet', markersize=3, linewidth=1 )
# a[0].plot([me,Me], [me,Me], 'g--',label = 'match', markersize=3, linewidth=1 )
# a[0].set_xlabel("Exact")
# a[0].set_ylabel("Numerical")
# a[0].legend(loc='upper left')
# a[0].grid(color='gray', linestyle='--', linewidth=0.5)

# me = min(n_eigenvalues[0:nmodes])
# Me = max(n_eigenvalues[0:nmodes])
# a[1].set_title(f"Neumann Eigenvalues {model.nx} x {model.ny}")
# a[1].plot(np.abs(n_eigenvalues[0:nmodes]), np.abs(nn_eigenvalues.cpu().numpy()),'o',label = 'neumann', markersize=3, linewidth=1 )
# a[1].plot([me,Me], [me,Me], 'g--',label = 'match', markersize=3, linewidth=1 )
# a[1].set_xlabel("Exact")
# a[1].set_ylabel("Numerical")
# a[1].legend(loc='upper left')
# a[1].grid(color='gray', linestyle='--', linewidth=0.5)

# plt.tight_layout()

# plt.savefig(f"eigenvalues-{model.nx}_{nmodes}.png")




for k in range(int(nmodes/6)):
  i0 = k*6
  f,a = plt.subplots(3,2)
  for j in range(6):
    im = a.flatten()[j].imshow(n_eigenvectors.cpu().numpy()[...,6*k+j].squeeze())
    f.colorbar(im, ax=a.flatten()[j],fraction=0.046,location='right')
    a.flatten()[j].set_title(f'e_{6*k+j}')

  plt.tight_layout()
  plt.savefig(f'neumann_modes_{k}.png')
  plt.close()

for k in range(int(nmodes/6)):
  i0 = k*6
  f,a = plt.subplots(3,2)
  for j in range(6):
    im = a.flatten()[j].imshow(d_eigenvectors.cpu().numpy()[...,6*k+j].squeeze())
    f.colorbar(im, ax=a.flatten()[j],fraction=0.046,location='right')
    a.flatten()[j].set_title(f'e_{6*k+j}')

  plt.tight_layout()
  plt.savefig(f'dirichlet_modes_{k}.png')
  plt.close()
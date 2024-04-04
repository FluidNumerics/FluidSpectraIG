#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
from fluidspectraig.mfeigen_torch import dot,norm
import time
from torch.profiler import profile, record_function, ProfilerActivity

nx = 256
ny = 256
Lx = 1.0#5120.0e3
Ly = 1.0#5120.0e3
nmodes = 100
nkrylov = 105
device = 'cuda'

# octogonal domain
mask = torch.ones(nx, ny)
for i in range(nx//4):
    for j in range(ny//4):
        if i+j < min(nx//4, ny//4):
            mask[i,j] = 0.
            mask[i,-1-j] = 0.
            mask[-1-i,j] = 0.
            mask[-1-i,-1-j] = 0.


param = {'nx': nx,
         'ny': ny,
         'Lx': Lx,
         'Ly': Ly,
         'device': device,
         'mask': mask,
         'dtype': torch.float64}

model = NMA(param)

f,a = plt.subplots(2,2)
im = a.flatten()[0].imshow(model.masks.q.squeeze().T.cpu().numpy(),interpolation='none')
f.colorbar(im, ax=a.flatten()[0],fraction=0.046,location='right')
a.flatten()[0].set_title('q')

im = a.flatten()[1].imshow(model.masks.u.squeeze().T.cpu().numpy(),interpolation='none')
f.colorbar(im, ax=a.flatten()[1],fraction=0.046,location='right')
a.flatten()[1].set_title('u')

im = a.flatten()[2].imshow(model.masks.v.squeeze().T.cpu().numpy(),interpolation='none')
f.colorbar(im, ax=a.flatten()[2],fraction=0.046,location='right')
a.flatten()[2].set_title('v')

im = a.flatten()[3].imshow(model.masks.psi.squeeze().T.cpu().numpy(),interpolation='none')
f.colorbar(im, ax=a.flatten()[3],fraction=0.046,location='right')
a.flatten()[3].set_title('psi')

plt.tight_layout()
plt.savefig('mask-check.png')
plt.close()


tic = time.perf_counter()
eigenvalues, eigenvectors, r, n_iters = model.calculate_dirichlet_modes(nmodes=nmodes,nkrylov=nkrylov,tol=1e-7, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s")
print(f"Dirichlet mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)


tic = time.perf_counter()
nn_eigenvalues, nn_eigenvectors, r, n_iters = model.calculate_neumann_modes(nmodes=nmodes,nkrylov=nkrylov,tol=1e-7, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Neumann mode runtime : {runtime} s")
print(f"Neumann mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

# # verify orthogonality
PtP = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
PtP_n = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
for row in range(nmodes):
  for col in range(nmodes):
    PtP[row,col] = dot( eigenvectors[...,row], eigenvectors[...,col] )
    PtP_n[row,col] = dot( nn_eigenvectors[...,row], nn_eigenvectors[...,col] )


f,a = plt.subplots(1,2)
im = a[0].imshow(PtP.cpu().numpy(),interpolation='none')
f.colorbar(im, ax=a[0],fraction=0.046,location='right')
a[0].set_title('dirichlet mode P^T P')

im = a[1].imshow(PtP_n.cpu().numpy(),interpolation='none')
f.colorbar(im, ax=a[1],fraction=0.046,location='right')
a[1].set_title('neumann mode P^T P')

plt.tight_layout()
plt.savefig('orthogonality-verification.png')
plt.close()



f,a = plt.subplots(3,2)
for k in range(6):
    v=nn_eigenvectors.cpu().numpy()[...,k].squeeze()
    v =np.ma.masked_where(model.masks.q.squeeze().cpu().numpy() == 0.0, v)
    im = a.flatten()[k].imshow(v,interpolation='none')
    f.colorbar(im, ax=a.flatten()[k],fraction=0.046,location='right')
    a.flatten()[k].set_title(f'e_{k}')

plt.tight_layout()
plt.savefig('neumann_modes_numerical.png')
plt.close()

f,a = plt.subplots(3,2)
for k in range(6):
    v=nn_eigenvectors.cpu().numpy()[...,nmodes-6+k].squeeze()
    v =np.ma.masked_where(model.masks.q.squeeze().cpu().numpy() == 0.0, v)
    im = a.flatten()[k].imshow(v,interpolation='none')
    f.colorbar(im, ax=a.flatten()[k],fraction=0.046,location='right')
    a.flatten()[k].set_title(f'e_{nmodes-6+k}')

plt.tight_layout()
plt.savefig('neumann_modes_numerical_small.png')
plt.close()

f,a = plt.subplots(3,2)
for k in range(6):
    v=eigenvectors.cpu().numpy()[...,k].squeeze()
    v =np.ma.masked_where(model.masks.psi.squeeze().cpu().numpy() == 0.0, v)
    im = a.flatten()[k].imshow(v,interpolation='none')
    f.colorbar(im, ax=a.flatten()[k],fraction=0.046,location='right')
    a.flatten()[k].set_title(f'e_{k}')
    
plt.tight_layout()
plt.savefig('dirichlet_modes_numerical.png')
plt.close()

f,a = plt.subplots(3,2)
for k in range(6):
    v=eigenvectors.cpu().numpy()[...,nmodes-6+k].squeeze()
    v =np.ma.masked_where(model.masks.psi.squeeze().cpu().numpy() == 0.0, v)
    im = a.flatten()[k].imshow(v,interpolation='none')
    f.colorbar(im, ax=a.flatten()[k],fraction=0.046,location='right')
    a.flatten()[k].set_title(f'e_{nmodes-6+k}')
    
plt.tight_layout()
plt.savefig('dirichlet_modes_numerical_small.png')
plt.close()



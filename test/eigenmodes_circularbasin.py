#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
from fluidspectraig.mfeigen_torch import dot
import time
from torch.profiler import profile, record_function, ProfilerActivity

nx = 192
ny = 240
Lx = 3840.0e3
Ly = 4800.0e3
device = 'cuda'

# vertex grid
xv = torch.linspace(0, Lx, nx+1, dtype=torch.float64, device=device)
yv = torch.linspace(0, Ly, ny+1, dtype=torch.float64, device=device)
x, y = torch.meshgrid(xv, yv, indexing='ij')

halfwidth = 0.125*Ly
center = 0.5*Ly
amplitude = 0.1*Lx
yc_percent = 0.5
mask = torch.ones(nx, ny)
for i in range(nx):
    for j in range(ny):
        shape = np.exp( -((yv[j].cpu()-yc_percent*Ly)**2)/(2.0*halfwidth**2) )
        if xv[i].cpu() < amplitude*shape:
            mask[i,j] = 0.


param = {'nx': nx,
         'ny': ny,
         'Lx': Lx,
         'Ly': Ly,
         'mask': mask,
         'device': device,
         'dtype': torch.float64}

model = NMA(param)
nmodes = 500

tic = time.perf_counter()
eigenvalues, eigenvectors, r, n_iters = model.calculate_dirichlet_modes(nmodes=500,tol=1e-7, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s")
print(f"Dirichlet mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

# tic = time.perf_counter()
# nn_eigenvalues, nn_eigenvectors, r, n_iters = model.calculate_neumann_modes(nmodes=500,tol=1e-7, max_iter=100)
# toc = time.perf_counter()

# runtime = toc - tic
# print(f"Neumann mode runtime : {runtime} s")
# print(f"Neumann mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

# verify orthogonality
PtP = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
#PtP_n = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
for row in range(nmodes):
  for col in range(nmodes):
    PtP[row,col] = dot( eigenvectors[...,row], eigenvectors[...,col] )
 #   PtP_n[row,col] = dot( nn_eigenvectors[...,row], nn_eigenvectors[...,col] )


f,a = plt.subplots(1,2)
im = a[0].imshow(PtP.cpu().numpy())
f.colorbar(im, ax=a[0],fraction=0.046,location='right')
a[0].set_title('dirichlet mode P^T P')

# im = a[1].imshow(PtP_n.cpu().numpy())
# f.colorbar(im, ax=a[1],fraction=0.046,location='right')
# a[1].set_title('neumann mode P^T P')

plt.tight_layout()
plt.savefig('orthogonality-verification.png')
plt.close()



# f,a = plt.subplots(3,2)
# im = a[0,0].imshow(nn_eigenvectors.cpu().numpy()[...,0].squeeze())
# f.colorbar(im, ax=a[0,0],fraction=0.046,location='right')
# a[0,0].set_title('e_0')

# im = a[0,1].imshow(nn_eigenvectors.cpu().numpy()[...,1].squeeze())
# f.colorbar(im, ax=a[0,1],fraction=0.046,location='right')
# a[0,1].set_title('e_1')

# im = a[1,0].imshow(nn_eigenvectors.cpu().numpy()[...,2].squeeze())
# f.colorbar(im, ax=a[1,0],fraction=0.046,location='right')
# a[1,0].set_title('e_2')

# im = a[1,1].imshow(nn_eigenvectors.cpu().numpy()[...,3].squeeze())
# f.colorbar(im, ax=a[1,1],fraction=0.046,location='right')
# a[1,1].set_title('e_3')

# im = a[2,0].imshow(nn_eigenvectors.cpu().numpy()[...,4].squeeze())
# f.colorbar(im, ax=a[2,0],fraction=0.046,location='right')
# a[2,0].set_title('e_4')

# im = a[2,1].imshow(nn_eigenvectors.cpu().numpy()[...,5].squeeze())
# f.colorbar(im, ax=a[2,1],fraction=0.046,location='right')
# a[2,1].set_title('e_5')

# plt.tight_layout()
# plt.savefig('neumann_modes_numerical.png')
# plt.close()

f,a = plt.subplots(3,2)
im = a[0,0].imshow(eigenvectors.cpu().numpy()[...,0].squeeze())
f.colorbar(im, ax=a[0,0],fraction=0.046,location='right')
a[0,0].set_title('e_0')

im = a[0,1].imshow(eigenvectors.cpu().numpy()[...,1].squeeze())
f.colorbar(im, ax=a[0,1],fraction=0.046,location='right')
a[0,1].set_title('e_1')

im = a[1,0].imshow(eigenvectors.cpu().numpy()[...,2].squeeze())
f.colorbar(im, ax=a[1,0],fraction=0.046,location='right')
a[1,0].set_title('e_2')

im = a[1,1].imshow(eigenvectors.cpu().numpy()[...,3].squeeze())
f.colorbar(im, ax=a[1,1],fraction=0.046,location='right')
a[1,1].set_title('e_3')

im = a[2,0].imshow(eigenvectors.cpu().numpy()[...,4].squeeze())
f.colorbar(im, ax=a[2,0],fraction=0.046,location='right')
a[2,0].set_title('e_4')

im = a[2,1].imshow(eigenvectors.cpu().numpy()[...,5].squeeze())
f.colorbar(im, ax=a[2,1],fraction=0.046,location='right')
a[2,1].set_title('e_5')

plt.tight_layout()
plt.savefig('dirichlet_modes_numerical.png')
plt.close()

# f,a = plt.subplots(1,1)
# im = a.imshow(H_square.cpu()[0:10,0:10].numpy())
# f.colorbar(im, ax=a,fraction=0.046,location='bottom')
# a.set_title('H_square')
# plt.tight_layout()
# plt.savefig('H_square.png')


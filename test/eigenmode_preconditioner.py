#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
from fluidspectraig.mfeigen_torch import dot
import time
from torch.profiler import profile, record_function, ProfilerActivity


param = {'nx': 100,
         'ny': 100,
         'Lx': 1.0,
         'Ly': 1.0,
         'nmodes':  20,
         'nkrylov': 25,
         'device': 'cuda',
         'dtype': torch.float64}

model = NMA(param)

tic = time.perf_counter()
eigenvalues, eigenvectors, r, n_iters = model.calculate_dirichlet_modes(tol=1e-7, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s")
print(f"Dirichlet mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

# Copy eigenmodes into structure to enable preconditioner
model.n_dirichlet = param['nmodes']
model.dirichlet_evals = eigenvalues
model.dirichlet_modes = eigenvectors

# Try solving an elliptic problem
fexact = torch.zeros_like(model.xg)
fexact = torch.sin(torch.pi*model.xg)*torch.sin(torch.pi*model.yg)
lapf = model.apply_laplacian_g(fexact)

tic = time.perf_counter()
#x = model.apply_laplacian_g_preconditioner(lapf).squeeze()
x = model.laplacian_g_inverse_pcg(lapf).squeeze()
toc = time.perf_counter()
runtime = toc - tic
print(f"Inverse runtime : {runtime} s")

f,a = plt.subplots(1,2)
um, uM = -1.1*np.abs(x.cpu().numpy()).max(), 1.1*np.abs(x.cpu().numpy()).max()
im = a[0].imshow(x.cpu().numpy().T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a[0],fraction=0.046,location='bottom')
a[0].set_title('f')

err = (x-fexact).cpu().numpy()
um, uM = -1.1*np.abs(err).max(), 1.1*np.abs(err).max()
im = a[1].imshow(err.T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a[1],fraction=0.046,location='bottom')
a[1].set_title('f-f_{exact}')

print(f"max error: {err.max()}")
plt.tight_layout()
plt.savefig('ftest_g.png')




# Attempt calculation again, with preconditioner.
# tic = time.perf_counter()
# eigenvalues, eigenvectors, r, n_iters = model.calculate_dirichlet_modes(tol=1e-7, max_iter=100)
# toc = time.perf_counter()

# runtime = toc - tic
# print(f"Dirichlet mode runtime : {runtime} s")
# print(f"Dirichlet mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

# tic = time.perf_counter()
# nn_eigenvalues, nn_eigenvectors, r, n_iters = model.calculate_neumann_modes(tol=1e-7, max_iter=100)
# toc = time.perf_counter()

# runtime = toc - tic
# print(f"Neumann mode runtime : {runtime} s")
# print(f"Neumann mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

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

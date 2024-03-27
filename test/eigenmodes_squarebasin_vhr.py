#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
import time


param = {'nx': 1000,
         'ny': 1000,
         'Lx': 1.0,
         'Ly': 1.0,
         'nmodes':  500,
         'nkrylov': 515,
         'device': 'cuda',
         'dtype': torch.float64}

model = NMA(param)

tic = time.perf_counter()
evals, evecs, r, n_iters = model.calculate_dirichlet_modes(tol=1e-9,max_iter=150)
toc = time.perf_counter()

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s", flush=True)
print(f"Dirichlet mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

f,a = plt.subplots(3,2)
im = a[0,0].imshow(evecs.cpu().numpy()[...,0].squeeze())
f.colorbar(im, ax=a[0,0],fraction=0.046,location='right')
a[0,0].set_title('e_0')

im = a[0,1].imshow(evecs.cpu().numpy()[...,1].squeeze())
f.colorbar(im, ax=a[0,1],fraction=0.046,location='right')
a[0,1].set_title('e_1')

im = a[1,0].imshow(evecs.cpu().numpy()[...,2].squeeze())
f.colorbar(im, ax=a[1,0],fraction=0.046,location='right')
a[1,0].set_title('e_2')

im = a[1,1].imshow(evecs.cpu().numpy()[...,3].squeeze())
f.colorbar(im, ax=a[1,1],fraction=0.046,location='right')
a[1,1].set_title('e_3')

im = a[2,0].imshow(evecs.cpu().numpy()[...,4].squeeze())
f.colorbar(im, ax=a[2,0],fraction=0.046,location='right')
a[2,0].set_title('e_4')

im = a[2,1].imshow(evecs.cpu().numpy()[...,5].squeeze())
f.colorbar(im, ax=a[2,1],fraction=0.046,location='right')
a[2,1].set_title('e_5')

plt.tight_layout()
plt.savefig(f"eigenmodes_numerical-{model.nx}x{model.ny}-{param['nmodes']}.png")
plt.close()

# f,a = plt.subplots(1,1)
# im = a.imshow(H_square.cpu()[0:10,0:10].numpy())
# f.colorbar(im, ax=a,fraction=0.046,location='bottom')
# a.set_title('H_square')
# plt.tight_layout()
# plt.savefig('H_square.png')


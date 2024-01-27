#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
import time

param = {'nx': 100,
         'ny': 100,
         'Lx': 1.0,
         'Ly': 0.1,
         'nmodes': 40,
         'nkrylov': 90,
         'device': 'cuda'}

model = NMA(param)

tic = time.perf_counter()
eigenvalues, eigenvectors, r = model.calculate_dirichlet_modes(tol=1e-12, max_iter=50)
toc = time.perf_counter()

print(eigenvalues.cpu().numpy())
    
print(f"residual : {r}")

f,a = plt.subplots(1,2)
im = a[0].imshow(eigenvectors.cpu().numpy()[...,0].squeeze())
f.colorbar(im, ax=a[0],fraction=0.046,location='bottom')
a[0].set_title('e_0')

im = a[1].imshow(eigenvectors.cpu().numpy()[...,1].squeeze())
f.colorbar(im, ax=a[1],fraction=0.046,location='bottom')
a[1].set_title('e_1')

plt.tight_layout()
plt.savefig('arnoldi.png')

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s")


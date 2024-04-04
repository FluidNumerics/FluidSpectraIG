#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
import time

param = {'nx': 400,
            'ny': 400,
            'Lx': 1.0,
            'Ly': 1.0,
            'nmodes': 10,
            'nkrylov': 12,
            'device': 'cuda'}

model = NMA(param)
fexact = torch.zeros_like(model.xc)
fexact = torch.cos(torch.pi*model.xc)*torch.cos(torch.pi*model.yc)

lapf = model.apply_laplacian_c(fexact)

tic = time.perf_counter()
x = model.laplacian_c_inverse_pcg(lapf).squeeze()
x = x - x.mean() # Adjust for the degenerate mode (enforce zero mean)
toc = time.perf_counter()
runtime = toc - tic
print(f"Inverse runtime : {runtime} s")

f,a = plt.subplots(1,2)
um, uM = -1.1*np.abs(x.cpu().numpy()).max(), 1.1*np.abs(x.cpu().numpy()).max()
im = a[0].imshow(x.cpu().numpy().T, cmap='bwr', origin='lower', vmin=um, vmax=uM,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a[0],fraction=0.046,location='bottom')
a[0].set_title('f')

err = (x-fexact).cpu().numpy()
um, uM = -1.1*np.abs(err).max(), 1.1*np.abs(err).max()
im = a[1].imshow(err.T, cmap='bwr', origin='lower',extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a[1],fraction=0.046,location='bottom')
a[1].set_title('f-f_{exact}')

print(f"max error: {err.max()}")
plt.tight_layout()
plt.savefig('ftest.png')

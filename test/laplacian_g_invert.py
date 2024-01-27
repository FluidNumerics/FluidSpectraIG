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
            'nmodes': 80,
            'device': 'cuda'}

model = NMA(param)
fexact = torch.zeros_like(model.xg)
fexact = torch.sin(torch.pi*model.xg)*torch.sin(torch.pi*model.yg)

b = model.apply_laplacian_g(fexact)#[...,1:-1,1:-1] # get b only at interior points


f,a = plt.subplots(1,1)
um, uM = -1.1*np.abs(b.cpu().numpy()).max(), 1.1*np.abs(b.cpu().numpy()).max()
im = a.imshow(b.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a,fraction=0.046,location='bottom')
a.set_title('b')
plt.tight_layout()
plt.savefig('b.png')

tic = time.perf_counter()
x = model.laplacian_g_inverse(b).squeeze()
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

plt.tight_layout()
plt.savefig('ftest.png')

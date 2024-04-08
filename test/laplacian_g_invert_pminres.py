#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA, laplacian_g
import time

param = {'nx': 100,
            'ny': 100,
            'Lx': 1.0,
            'Ly': 1.0,
            'nmodes': 10,
            'nkrylov': 12,
            'device': 'cuda'}

model = NMA(param)
fexact = torch.zeros_like(model.xg)
fexact = torch.sin(torch.pi*model.xg)*torch.sin(torch.pi*model.yg)
print(f"f shape   {fexact.cpu().numpy().T.squeeze().shape}")



lapf = model.apply_laplacian_g(fexact)

f,a = plt.subplots(1,2)

im = a.flatten()[0].imshow(fexact.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[0],fraction=0.046,location='bottom')
a.flatten()[0].set_title('f')

print(f"L(f) shape   {lapf.cpu().numpy().T.squeeze().shape}")
im = a.flatten()[1].imshow(lapf.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[1],fraction=0.046,location='bottom')
a.flatten()[1].set_title('L(f)')

plt.tight_layout()
plt.savefig('f_lapf.png')
plt.close()

f,a = plt.subplots(2,2)



tic = time.perf_counter()
x = model.laplacian_g_inverse_pminres(lapf).squeeze()
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

#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from nma import NMA

param = {'nx': 100,
            'ny': 100,
            'Lx': 1.0,
            'Ly': 1.0,
            'nmodes': 80,
            'device': 'cuda'}

model = NMA(param)
fexact = torch.zeros_like(model.xg)
fexact = torch.sin(torch.pi*model.xg)*torch.sin(torch.pi*model.yg)

b = model.apply_laplacian_g(fexact)

f,a = plt.subplots(1,1)
um, uM = -1.1*np.abs(b.cpu().numpy()).max(), 1.1*np.abs(b.cpu().numpy()).max()
im = a.imshow(b.cpu().numpy().T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a,fraction=0.046,location='bottom')
a.set_title('b')
plt.tight_layout()
plt.savefig('b.png')

f0 = torch.zeros_like(model.xg)
fm = model.laplacian_g_inverse(b,f0)

f,a = plt.subplots(1,2)
um, uM = -1.1*np.abs(fm.cpu().numpy()).max(), 1.1*np.abs(fm.cpu().numpy()).max()
im = a[0].imshow(fm.cpu().numpy().T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a[0],fraction=0.046,location='bottom')
a[0].set_title('f')

err = (fm-fexact).cpu().numpy()
um, uM = -1.1*np.abs(err).max(), 1.1*np.abs(err).max()
im = a[1].imshow(err.T, cmap='bwr', origin='lower', vmin=um, vmax=uM, animated=True,extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a[1],fraction=0.046,location='bottom')
a[1].set_title('f-f_{exact}')

plt.tight_layout()
plt.savefig('ftest.png')

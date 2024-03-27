#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA, dfdx_c, dfdy_c, laplacian_c
import time

param = {'nx': 100,
            'ny': 100,
            'Lx': 1.0,
            'Ly': 1.0,
            'nmodes': 10,
            'nkrylov': 12,
            'device': 'cuda'}

model = NMA(param)
fexact = torch.zeros_like(model.xc)
fexact = torch.cos(torch.pi*model.xc)*torch.cos(torch.pi*model.yc)
print(f"f shape   {fexact.cpu().numpy().T.squeeze().shape}")


dfdx = dfdx_c(fexact,model.dx)
dfdy = dfdy_c(fexact,model.dy)
lapf = laplacian_c(fexact,model.dx,model.dy)

f,a = plt.subplots(2,2)

print(f"dfdx shape   {dfdx.cpu().numpy().T.squeeze().shape}")
im = a.flatten()[0].imshow(dfdx.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[0],fraction=0.046,location='bottom')
a.flatten()[0].set_title('dfdx')

print(f"dfdy shape   {dfdy.cpu().numpy().T.squeeze().shape}")
im = a.flatten()[1].imshow(dfdy.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[1],fraction=0.046,location='bottom')
a.flatten()[1].set_title('dfdy')

im = a.flatten()[2].imshow(fexact.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[2],fraction=0.046,location='bottom')
a.flatten()[2].set_title('f')

print(f"L(f) shape   {lapf.cpu().numpy().T.squeeze().shape}")
im = a.flatten()[3].imshow(lapf.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[3],fraction=0.046,location='bottom')
a.flatten()[3].set_title('L(f)')

plt.tight_layout()
plt.savefig('gradf.png')
plt.close()

f,a = plt.subplots(2,2)

print(f"q mask shape   {model.masks.q.cpu().numpy().T.squeeze().shape}")
im = a.flatten()[0].imshow(model.masks.q.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[0],fraction=0.046,location='bottom')
a.flatten()[0].set_title('q')

im = a.flatten()[1].imshow(model.masks.psi.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[1],fraction=0.046,location='bottom')
a.flatten()[1].set_title('psi')
print(f"psi mask shape {model.masks.psi.cpu().numpy().T.squeeze().shape}")


im = a.flatten()[2].imshow(model.masks.u.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[2],fraction=0.046,location='bottom')
a.flatten()[2].set_title('u')
print(f"u mask shape   {model.masks.u.cpu().numpy().T.squeeze().shape}")

im = a.flatten()[3].imshow(model.masks.v.cpu().numpy().T.squeeze(), cmap='bwr', origin='lower', extent=[0,param['Lx'],0,param['Ly']], aspect=param['Lx']/param['Ly'])
f.colorbar(im, ax=a.flatten()[3],fraction=0.046,location='bottom')
a.flatten()[3].set_title('v')
print(f"v mask shape   {model.masks.v.cpu().numpy().T.squeeze().shape}")

plt.tight_layout()
plt.savefig('masks.png')


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

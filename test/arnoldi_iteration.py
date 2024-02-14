#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
import time

def DirichletModes( model ):
    """ Calculates the exact dirichlet modes for a rectangular domain"""

    nx = int((model.nx))
    ny = int((model.ny))
    Lx = model.Lx
    Ly = model.Ly
    nmodes = nx*ny
    eigenmodes = np.zeros((nmodes,nx+1,ny+1))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nx)
    for m in range(0, nx):
      kx[m] = (m+1)*np.pi/Lx
        
    ky = np.zeros(ny)
    for m in range(0, ny):
      ky[m] = (m+1)*np.pi/Ly

    k = 0
    tmp = np.zeros((nmodes,nx+1,ny+1))
    ev = np.zeros(nmodes)
    xg = model.xg.cpu().numpy()
    yg = model.yg.cpu().numpy()
    for m in range(0, ny):
        for n in range(0, nx):
            x = xg*kx[n]
            y = yg*ky[m]
            tmp[k,:,:] = np.sin( x )*np.sin( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]
    for k in range(0,nmodes):
        eigenmodes[k,:,:] = tmp[sort_index[k],:,:].squeeze()

    return eigenvalues, eigenmodes


param = {'nx': 100,
         'ny': 100,
         'Lx': 1.0,
         'Ly': 1.0,
         'nmodes': 20,
         'nkrylov': 40,
         'device': 'cuda'}

model = NMA(param)

tic = time.perf_counter()
eigenvalues, eigenvectors, r = model.calculate_dirichlet_modes(tol=1e-8, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s")


d_eigenvalues, d_eigenvectors = DirichletModes(model)

plt.figure()
plt.title(f"Eigenvalues {model.nx} x {model.ny}")
plt.plot(np.abs(d_eigenvalues[0:param['nmodes']]),'-o',label = 'dirichlet (exact)', markersize=3, linewidth=1 )
plt.plot(np.abs(eigenvalues.cpu().numpy()),'-x',label = 'dirichlet (numerical)', markersize=4, linewidth=1 )
plt.legend(loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.savefig(f"eigenvalues-{model.nx}_{param['nmodes']}.png")


f,a = plt.subplots(3,2)
im = a[0,0].imshow(d_eigenvectors[0,...].squeeze())
f.colorbar(im, ax=a[0,0],fraction=0.046,location='bottom')
a[0,0].set_title('e_0')

im = a[0,1].imshow(d_eigenvectors[1,...].squeeze())
f.colorbar(im, ax=a[0,1],fraction=0.046,location='bottom')
a[0,1].set_title('e_1')

im = a[1,0].imshow(d_eigenvectors[2,...].squeeze())
f.colorbar(im, ax=a[1,0],fraction=0.046,location='bottom')
a[1,0].set_title('e_2')

im = a[1,1].imshow(d_eigenvectors[3,...].squeeze())
f.colorbar(im, ax=a[1,1],fraction=0.046,location='bottom')
a[1,1].set_title('e_3')

im = a[2,0].imshow(d_eigenvectors[4,...].squeeze())
f.colorbar(im, ax=a[2,0],fraction=0.046,location='bottom')
a[2,0].set_title('e_4')

im = a[2,1].imshow(d_eigenvectors[5,...].squeeze())
f.colorbar(im, ax=a[2,1],fraction=0.046,location='bottom')
a[2,1].set_title('e_5')

plt.tight_layout()
plt.savefig('exact.png')
plt.close()



f,a = plt.subplots(3,2)
im = a[0,0].imshow(eigenvectors.cpu().numpy()[...,0].squeeze())
f.colorbar(im, ax=a[0,0],fraction=0.046,location='bottom')
a[0,0].set_title('e_0')

im = a[0,1].imshow(eigenvectors.cpu().numpy()[...,1].squeeze())
f.colorbar(im, ax=a[0,1],fraction=0.046,location='bottom')
a[0,1].set_title('e_1')

im = a[1,0].imshow(eigenvectors.cpu().numpy()[...,2].squeeze())
f.colorbar(im, ax=a[1,0],fraction=0.046,location='bottom')
a[1,0].set_title('e_2')

im = a[1,1].imshow(eigenvectors.cpu().numpy()[...,3].squeeze())
f.colorbar(im, ax=a[1,1],fraction=0.046,location='bottom')
a[1,1].set_title('e_3')

im = a[2,0].imshow(eigenvectors.cpu().numpy()[...,4].squeeze())
f.colorbar(im, ax=a[2,0],fraction=0.046,location='bottom')
a[2,0].set_title('e_4')

im = a[2,1].imshow(eigenvectors.cpu().numpy()[...,5].squeeze())
f.colorbar(im, ax=a[2,1],fraction=0.046,location='bottom')
a[2,1].set_title('e_5')

plt.tight_layout()
plt.savefig('arnoldi.png')
plt.close()

# f,a = plt.subplots(1,1)
# im = a.imshow(H_square.cpu()[0:10,0:10].numpy())
# f.colorbar(im, ax=a,fraction=0.046,location='bottom')
# a.set_title('H_square')
# plt.tight_layout()
# plt.savefig('H_square.png')


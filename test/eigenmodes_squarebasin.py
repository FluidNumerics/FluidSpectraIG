#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from fluidspectraig.nma import NMA
from fluidspectraig.mfeigen_torch import dot
import time
from torch.profiler import profile, record_function, ProfilerActivity

def DirichletModes( model ):
    """ Calculates the exact dirichlet modes for a rectangular domain"""

    nx = int((model.nx))
    ny = int((model.ny))
    Lx = model.Lx
    Ly = model.Ly
    nmodes = nx*ny
    #eigenmodes = np.zeros((nmodes,nx+1,ny+1))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nx)
    for m in range(0, nx):
      kx[m] = (m+1)*np.pi/Lx
        
    ky = np.zeros(ny)
    for m in range(0, ny):
      ky[m] = (m+1)*np.pi/Ly

    k = 0
    #tmp = np.zeros((nmodes,nx+1,ny+1))
    ev = np.zeros(nmodes)
    xg = model.xg.cpu().numpy()
    yg = model.yg.cpu().numpy()
    for m in range(0, ny):
        for n in range(0, nx):
            x = xg*kx[n]
            y = yg*ky[m]
            #tmp[k,:,:] = np.sin( x )*np.sin( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]
    #for k in range(0,nmodes):
    #    eigenmodes[k,:,:] = tmp[sort_index[k],:,:].squeeze()

    return eigenvalues#, eigenmodes

def NeumannModes( model ):
    """ Calculates the exact neumann modes for a rectangular domain"""

    nx = int((model.nx))
    ny = int((model.ny))
    Lx = model.Lx
    Ly = model.Ly
    nmodes = nx*ny
    #eigenmodes = np.zeros((nmodes,nx+1,ny+1))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nx)
    for m in range(0, nx):
      kx[m] = (m)*np.pi/Lx
        
    ky = np.zeros(ny)
    for m in range(0, ny):
      ky[m] = (m)*np.pi/Ly

    k = 0
    #tmp = np.zeros((nmodes,nx+1,ny+1))
    ev = np.zeros(nmodes)
    xg = model.xg.cpu().numpy()
    yg = model.yg.cpu().numpy()
    for m in range(0, ny):
        for n in range(0, nx):
            x = xg*kx[n]
            y = yg*ky[m]
          #  tmp[k,:,:] = np.cos( x )*np.cos( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]
    #for k in range(0,nmodes):
    #    eigenmodes[k,:,:] = tmp[sort_index[k],:,:].squeeze()

    return eigenvalues #, eigenmodes


param = {'nx': 250,
         'ny': 250,
         'Lx': 1.0,
         'Ly': 1.0,
         'device': 'cuda',
         'dtype': torch.float64}

model = NMA(param)
nmodes = 500

tic = time.perf_counter()
eigenvalues, eigenvectors, r, n_iters = model.calculate_dirichlet_modes(nmodes=500,tol=1e-7, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Dirichlet mode runtime : {runtime} s")
print(f"Dirichlet mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

tic = time.perf_counter()
nn_eigenvalues, nn_eigenvectors, r, n_iters = model.calculate_neumann_modes(nmodes=500,tol=1e-7, max_iter=100)
toc = time.perf_counter()

runtime = toc - tic
print(f"Neumann mode runtime : {runtime} s")
print(f"Neumann mode runtime per iterate : {runtime/n_iters} [s/iterate]", flush=True)

# verify orthogonality
PtP = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
PtP_n = torch.zeros(nmodes,nmodes, **model.arr_kwargs)
for row in range(nmodes):
  for col in range(nmodes):
    PtP[row,col] = dot( eigenvectors[...,row], eigenvectors[...,col] )
    PtP_n[row,col] = dot( nn_eigenvectors[...,row], nn_eigenvectors[...,col] )


f,a = plt.subplots(1,2)
im = a[0].imshow(PtP.cpu().numpy())
f.colorbar(im, ax=a[0],fraction=0.046,location='right')
a[0].set_title('dirichlet mode P^T P')

im = a[1].imshow(PtP_n.cpu().numpy())
f.colorbar(im, ax=a[1],fraction=0.046,location='right')
a[1].set_title('neumann mode P^T P')

plt.tight_layout()
plt.savefig('orthogonality-verification.png')
plt.close()

d_eigenvalues = DirichletModes(model)
n_eigenvalues = NeumannModes(model)

f,a = plt.subplots(1,2)
me = min(d_eigenvalues[0:nmodes])
Me = max(d_eigenvalues[0:nmodes])
a[0].set_title(f"Dirichlet Eigenvalues {model.nx} x {model.ny}")
a[0].plot(np.abs(d_eigenvalues[0:nmodes]), np.abs(eigenvalues.cpu().numpy()),'o',label = 'dirichlet', markersize=3, linewidth=1 )
a[0].plot([me,Me], [me,Me], 'g--',label = 'match', markersize=3, linewidth=1 )
a[0].set_xlabel("Exact")
a[0].set_ylabel("Numerical")
a[0].legend(loc='upper left')
a[0].grid(color='gray', linestyle='--', linewidth=0.5)

me = min(n_eigenvalues[0:nmodes])
Me = max(n_eigenvalues[0:nmodes])
a[1].set_title(f"Neumann Eigenvalues {model.nx} x {model.ny}")
a[1].plot(np.abs(n_eigenvalues[0:nmodes]), np.abs(nn_eigenvalues.cpu().numpy()),'o',label = 'neumann', markersize=3, linewidth=1 )
a[1].plot([me,Me], [me,Me], 'g--',label = 'match', markersize=3, linewidth=1 )
a[1].set_xlabel("Exact")
a[1].set_ylabel("Numerical")
a[1].legend(loc='upper left')
a[1].grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.savefig(f"eigenvalues-{model.nx}_{nmodes}.png")


# f,a = plt.subplots(3,2)
# im = a[0,0].imshow(d_eigenvectors[0,...].squeeze())
# f.colorbar(im, ax=a[0,0],fraction=0.046,location='right')
# a[0,0].set_title('e_0')

# im = a[0,1].imshow(d_eigenvectors[1,...].squeeze())
# f.colorbar(im, ax=a[0,1],fraction=0.046,location='right')
# a[0,1].set_title('e_1')

# im = a[1,0].imshow(d_eigenvectors[2,...].squeeze())
# f.colorbar(im, ax=a[1,0],fraction=0.046,location='right')
# a[1,0].set_title('e_2')

# im = a[1,1].imshow(d_eigenvectors[3,...].squeeze())
# f.colorbar(im, ax=a[1,1],fraction=0.046,location='right')
# a[1,1].set_title('e_3')

# im = a[2,0].imshow(d_eigenvectors[4,...].squeeze())
# f.colorbar(im, ax=a[2,0],fraction=0.046,location='right')
# a[2,0].set_title('e_4')

# im = a[2,1].imshow(d_eigenvectors[5,...].squeeze())
# f.colorbar(im, ax=a[2,1],fraction=0.046,location='right')
# a[2,1].set_title('e_5')

# plt.tight_layout()
# plt.savefig('eigenmodes_exact.png')
# plt.close()



f,a = plt.subplots(3,2)
im = a[0,0].imshow(nn_eigenvectors.cpu().numpy()[...,0].squeeze())
f.colorbar(im, ax=a[0,0],fraction=0.046,location='right')
a[0,0].set_title('e_0')

im = a[0,1].imshow(nn_eigenvectors.cpu().numpy()[...,1].squeeze())
f.colorbar(im, ax=a[0,1],fraction=0.046,location='right')
a[0,1].set_title('e_1')

im = a[1,0].imshow(nn_eigenvectors.cpu().numpy()[...,2].squeeze())
f.colorbar(im, ax=a[1,0],fraction=0.046,location='right')
a[1,0].set_title('e_2')

im = a[1,1].imshow(nn_eigenvectors.cpu().numpy()[...,3].squeeze())
f.colorbar(im, ax=a[1,1],fraction=0.046,location='right')
a[1,1].set_title('e_3')

im = a[2,0].imshow(nn_eigenvectors.cpu().numpy()[...,4].squeeze())
f.colorbar(im, ax=a[2,0],fraction=0.046,location='right')
a[2,0].set_title('e_4')

im = a[2,1].imshow(nn_eigenvectors.cpu().numpy()[...,5].squeeze())
f.colorbar(im, ax=a[2,1],fraction=0.046,location='right')
a[2,1].set_title('e_5')

plt.tight_layout()
plt.savefig('neumann_modes_numerical.png')
plt.close()

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


#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from fluidspectraig.tuml import TUML
from fluidspectraig.splig import splig
from fluidspectraig.nma import NMA
import os
import pickle 

nx = 64
ny = 64

case_dir = f"square_domain/{nx}x{ny}"
if not os.path.exists(case_dir):
    os.makedirs(case_dir)

# Create the model
param = {'nx': nx,
         'ny': ny,
         'Lx': 1.0,
         'Ly': 1.0,
         'case_directory': case_dir,
         'device': 'cpu',
         'dtype': torch.float64}

nma_obj = NMA(param,model=TUML)
nma_obj.construct_splig()
nma_obj.write() # Save the nma_obj to disk in the case directory

# Save a few plots for reference
# Plot the mask
plt.figure()
plt.imshow(nma_obj.mask_d,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/dirichlet-mask.png')
plt.close()

plt.figure()
plt.imshow(nma_obj.mask_n,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/neumann-mask.png')
plt.close()

## [TO DO]
# 
# Generate instructions for computing eigenpairs using the provided binary #
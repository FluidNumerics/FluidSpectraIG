#!/usr/bin/env python
# 
# This example is meant to show a complete walkthrough for computing
# the dirichlet and neumann modes for the wind-driven gyre example from
# L. Thiry's MQGeometry.
#
# Once the sparse matrices are created with this script, the dirichlet
# and neumann mode eigenpairs can be diagnosed with ../bin/laplacian_modes
#
# From here, the eigenmodes and eigenvalues can be used to calcualte the spectra 
# of the velocity field obtained with a QG simulation from MQGeometry.
# 
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from fluidspectraig.mitgcm import MITgcm
from fluidspectraig.nma import NMA
import os


case_dir = "mitgcm-100-z75"
model_dir = "mitgcm-100-z75/production"
if not os.path.exists(case_dir):
    os.makedirs(case_dir)

# Create the model
param = {'case_directory': case_dir,
         'model_directory': model_dir,
         'device': 'cpu',
         'dtype': torch.float64}

nma_obj = NMA(param,model=MITgcm)

#nma_obj.construct_splig()
#nma_obj.write() # Save the nma_obj to disk in the case directory



# Save a few plots for reference
# Plot the mask
plt.figure()
mask_d = nma_obj.model.masks.psi.type(torch.int32).squeeze().cpu().numpy()
plt.imshow(mask_d,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/dirichlet-mask.png')
plt.close()

plt.figure()
mask_n = nma_obj.model.masks.q.type(torch.int32).squeeze().cpu().numpy()
plt.imshow(mask_n,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/neumann-mask.png')
plt.close()

## [TO DO]
# 
# Generate instructions for computing eigenpairs using the provided binary #
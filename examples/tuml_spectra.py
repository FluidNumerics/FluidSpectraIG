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
from fluidspectraig.nma import NMA, load_param
from fluidspectraig.tuml import TUML
import os
import sys

plt.style.use('seaborn-v0_8-whitegrid')
plt.switch_backend('agg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

case_dir = "square_domain/64x64/"

# Load parameters
param = load_param(case_dir)

# Initialize the model
nma_obj = NMA(param,model=TUML)

# Load the case directory
nma_obj.load(case_dir)

u = torch.ones((nma_obj.splig_d.nx,nma_obj.splig_n.ny),**nma_obj.arr_kwargs)
v = torch.zeros((nma_obj.splig_n.nx,nma_obj.splig_d.ny),**nma_obj.arr_kwargs)

print(f"shape(u) : {u.shape}")
print(f"shape(v) : {v.shape}")

#nma_obj.plot_eigenmodes()

lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m = nma_obj.spectra(u,v,atol=1e-5)

total_energy = np.sum(Edi_m) + np.sum(Edb_m) + np.sum(Eri_m) + np.sum(Erb_m)
print(f" Estimated total energy : {total_energy:.3E} (m^4/s^2)")
print(f" Estimated energy density : {total_energy/nma_obj.model.total_area_n():.3E} (m^2/s^2)")

energy_components = ["Rotational (interior)","Rotational (boundary)","Divergent (interior)","Divergent (boundary)"]
energy_values = [np.sum(Eri_m),np.sum(Erb_m),np.sum(Edi_m),np.sum(Edb_m)]
y_pos = np.arange(len(energy_components))

xU = max(energy_values)*1.25
fig,ax = plt.subplots(1,1)
ax.barh(y_pos,energy_values)
ax.set_yticks(y_pos, labels=energy_components) 
ax.set_xlabel("Energy ($m^4 s^{-2}$)")
plt.xlim(0, xU)
for y in y_pos:
    if energy_values[y] < 0.25*xU:
        ax.text(energy_values[y]+0.01*xU, y, f'{energy_values[y]:.3f} ', color='black', ha='left', va='center')
    else:
        ax.text(energy_values[y]-0.01*xU, y, f'{energy_values[y]:.3f} ', color='white', ha='right', va='center')

plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
plt.savefig(f"{case_dir}/constantvelocity_total_energy_budget.png")
plt.close()

wavenumber = 2.0*np.pi*np.sqrt(lambda_m)
plt.figure
# neumann mode - divergent component
plt.plot( wavenumber, Edi_m/nma_obj.model.total_area_n(), label="Interior" )
plt.plot( wavenumber, Edb_m/nma_obj.model.total_area_n(), label="Boundary" )
plt.title("Divergent Spectra")
plt.xlabel("wavenumber (rad/m)")
plt.ylabel("E ($m^2 s^{-2}$)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig(f"{case_dir}/constantvelocity_divergent_spectra.png")
plt.close()

wavenumber = 2.0*np.pi*np.sqrt(sigma_m)
plt.figure
# dirichlet mode - rotational component
plt.plot( wavenumber, Eri_m/nma_obj.model.total_area_n(), label="Interior")
plt.plot( wavenumber, Erb_m/nma_obj.model.total_area_n(), label="Boundary" )
plt.title("Rotational Spectra")
plt.xlabel("wavenumber (rad/m)")
plt.ylabel("E ($m^2 s^{-2}$)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig(f"{case_dir}/constantvelocity_rotational_spectra.png")

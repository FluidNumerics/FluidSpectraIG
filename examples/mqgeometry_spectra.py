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
from fluidspectraig.tuml import TUML, grad_perp
import os
import sys

plt.style.use('seaborn-v0_8-whitegrid')
plt.switch_backend('agg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

force_recompute = True

case_dir = "mqgeometry_doublegyre-squarebasin/256x256"
model_input = f"{case_dir}/mqgeometry-dt3600/psi_100y_000d.npy"

# Set up output file name for spectra
model_directory = "/".join(model_input.split("/")[:-1])
model_file_basename = model_input.split("/")[-1].split(".")[0]
spectra_output_file = f"{model_directory}/{model_file_basename}_spectra.npz"

print(f"Output spectra : {spectra_output_file}")

# Load parameters
param = load_param(case_dir)
nma_obj = NMA(param,model=TUML)
nma_obj.load(case_dir)

# Load the MQGeometry stream function from .npy output
print(f"Loading model input : {model_input}")
psi = torch.from_numpy(np.load(model_input)[0,0,:,:].squeeze())

# Calculate the velocity field
print("Computing velocity from geostrophic stream function")
u, v = grad_perp(psi,nma_obj.model.dx,nma_obj.model.dy)

print(f"min(u), max(u) : {torch.min(u)}, {torch.max(u)}")
print(f"min(v), max(v) : {torch.min(v)}, {torch.max(v)}")

#nma_obj.plot_eigenmodes()

lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m = nma_obj.spectra(u,v,atol=1e-5)
if (not os.path.exists(spectra_output_file)) or force_recompute:
    print("Computing Spectra")
    lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m = nma_obj.spectra(u,v)

    np.savez( spectra_output_file, lambda_m = lambda_m, sigma_m = sigma_m,
        Edi_m = Edi_m, Eri_m = Eri_m, Edb_m = Edb_m, Erb_m = Erb_m )
else:
    print(f"Spectra output found from previous computation. Using file {spectra_output_file}")
    with np.load(spectra_output_file) as data:
        lambda_m = data['lambda_m']*1e-6
        sigma_m = data['sigma_m']*1e-6
        Edi_m = data['Edi_m']
        Eri_m = data['Eri_m']
        Edb_m = data['Edb_m']
        Erb_m = data['Erb_m']



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
        ax.text(energy_values[y]+0.01*xU, y, f'{energy_values[y]:.3E} ', color='black', ha='left', va='center')
    else:
        ax.text(energy_values[y]-0.01*xU, y, f'{energy_values[y]:.3E} ', color='white', ha='right', va='center')

plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
plt.savefig(f"{model_input}.total_energy_budget.png")
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
plt.savefig(f"{model_input}.divergent_spectra.png")
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
plt.savefig(f"{model_input}.rotational_spectra.png")
plt.close()


# #!/usr/bin/env python
# # 
# # This example is meant to show a complete walkthrough for computing
# # the dirichlet and neumann modes for the wind-driven gyre example from
# # L. Thiry's MQGeometry.
# #
# # Once the sparse matrices are created with this script, the dirichlet
# # and neumann mode eigenpairs can be diagnosed with ../bin/laplacian_modes
# #
# # From here, the eigenmodes and eigenvalues can be used to calcualte the spectra 
# # of the velocity field obtained with a QG simulation from MQGeometry.
# # 
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import numpy.ma as ma
# from fluidspectraig.splig import splig
# from fluidspectraig.nma import NMA
# import os

# plt.style.use('seaborn-v0_8-whitegrid')
# plt.switch_backend('agg')


# #nma_obj.plot_eigenmodes()



# wavenumber = 2.0*np.pi*np.sqrt(lambda_m)
# print(f" Estimated energy : {np.sum(Eri_m)} (m^4/s^2)")
# plt.figure
# # neumann mode - divergent component
# plt.plot( wavenumber, Edi_m, label="Interior" )
# plt.plot( wavenumber, Edb_m, label="Boundary" )
# plt.title("Divergent Spectra")
# plt.xlabel("wavenumber (rad/m)")
# plt.ylabel("E")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid(True, which="both", ls="-", color='0.65')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# plt.savefig(f"{model_input}.divergent_spectra.png")
# plt.close()

# wavenumber = 2.0*np.pi*np.sqrt(sigma_m)
# plt.figure
# # dirichlet mode - rotational component
# plt.plot( wavenumber, Eri_m, label="Interior")
# plt.plot( wavenumber, Erb_m, label="Boundary" )
# plt.title("Rotational Spectra")
# plt.xlabel("wavenumber (rad/m)")
# plt.ylabel("E")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid(True, which="both", ls="-", color='0.65')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# plt.savefig(f"{model_input}.rotational_spectra.png")


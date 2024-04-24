

import numpy as np
import torch
from fluidspectraig.splig import splig, splig_load
import h5py

def norm(u,mask):
    """Calculates the magnitude of grid data"""
    return  torch.sqrt( torch.sum(u*u*mask) )

def dot(u,v,mask):
    """Performs dot product on grid data"""
    return torch.sum( u*v*mask )

class NMA:
    """Normal Mode Analysis class"""
    def __init__(self):
        self.initialized = True

        self.splig_d = None
        self.evec_d = None
        self.eval_d = None

        self.splig_n = None
        self.evec_n = None
        self.eval_n = None

        self.case_directory = "./"


    def load(self, case_directory):

        def Filter(string, substr):
            return [str for str in string if any(sub in str for sub in substr)]

        self.case_directory = case_directory
        self.splig_d = splig_load(f"{case_directory}/dirichlet")
        print(f"Loading dirichlet mode eigenvectors from {case_directory}/dirichlet.evec.hdf5")
        self.evec_d = h5py.File(f"{case_directory}/dirichlet.evec.hdf5",'r')
        print(f"Loading dirichlet mode eigenvalues from {case_directory}/dirichlet.eval.hdf5")
        fobj = h5py.File(f"{case_directory}/dirichlet.eval.hdf5",'r')
        obj_key = Filter(fobj.keys(),['eigr'])[0]
        self.eval_d = fobj[obj_key]


        # Get the keys for only the real components of the eigenvectors
        self.evec_d_rkeys = Filter(self.evec_d.keys(),['Xr'])
        self.evec_d_tag = "_".join(self.evec_d_rkeys[0].split("_")[1:])

        self.neval_d = int(len(self.evec_d_rkeys))
        print(f"Number of dirichlet eigenmodes : {self.neval_d}")
        print("")

        self.splig_n = splig_load(f"{case_directory}/neumann")
        print(f"Loading neumann mode eigenvectors from {case_directory}/neumann.evec.hdf5")
        self.evec_n = h5py.File(f"{case_directory}/neumann.evec.hdf5",'r')
        print(f"Loading neumann mode eigenvalues from {case_directory}/neumann.eval.hdf5")
        fobj = h5py.File(f"{case_directory}/neumann.eval.hdf5",'r')
        obj_key = Filter(fobj.keys(),['eigr'])[0]
        self.eval_n = fobj[obj_key]

        # Get the keys for only the real components of the eigenvectors
        self.evec_n_rkeys = Filter(self.evec_n.keys(),['Xr'])
        self.evec_n_tag = "_".join(self.evec_n_rkeys[0].split("_")[1:])

        self.neval_n = int(len(self.evec_n_rkeys))
        print(f"Number of neumann eigenmodes : {self.neval_n}")

    def get_dirichlet_mode(self,k):
        import numpy as np
        import numpy.ma as ma

        if k < self.neval_d:
            obj_key = f"Xr{k}_{self.evec_d_tag}"
            if obj_key in list(self.evec_d_rkeys):
                v = self.evec_d[obj_key]
                v_gridded = ma.array( np.zeros((self.splig_d.nx,self.splig_d.ny)), dtype=np.float64, order='C', mask=np.abs(self.splig_d.mask-1),fill_value=0.0 )
                v_gridded[~v_gridded.mask] = v
                return v_gridded
            else: 
                print(f"{obj_key} not found in dirichlet eigenvectors h5 index.")
        else:
            print(f"{k} exceeds number of dirichlet modes {self.neval_d}")
            return None

    def get_neumann_mode(self,k):
        import numpy as np
        import numpy.ma as ma

        if k < self.neval_n:
            obj_key = f"Xr{k}_{self.evec_n_tag}"
            if obj_key in list(self.evec_n_rkeys):
                v = self.evec_n[obj_key]
                v_gridded = ma.array( np.zeros((self.splig_n.nx,self.splig_n.ny)), dtype=np.float64, order='C', mask=np.abs(self.splig_n.mask-1),fill_value=0.0 )
                v_gridded[~v_gridded.mask] = v
                return v_gridded
            else: 
                print(f"{obj_key} not found in neumann eigenvectors h5 index.")
        else:
            print(f"{k} exceeds number of neumann modes {self.neval_n}")
            return None

    def spectra(self, u, v, decimals=9):
        """Calculates the energy spectra for a velocity field (u,v).
        
        This routine calculates the following projection coefficiens

            di_m - Divergent (Neumann) mode projection coefficients, interior component
            db_m - Dirichlet (Neumann) mode projection coefficients, boundary component
            vi_m - Vorticity (Dirichlet) mode projection coefficients, interior component
            vb_m - Vorticity (Dirichlet) mode projection coefficients, interior component

        The energy is broken down into four parts

            1. Divergent interior
            2. Rotational interior
            3. Divergent boundary
            4. Rotational boundary
        
        Each component is defined as

            1. Edi_{m} = -0.5*di_m*di_m/\lambda_m 
            2. Eri_{m} = -0.5*vi_m*vi_m/\sigma_m 
            3. Edb_{m} = -(0.5*db_m*db_m + db_m*di_m)/\lambda_m 
            4. Erb_{m} = -(0.5*vb_m*vb_m + vb_m*vi_m)/\sigma_m         

        Once calculated, the spectra is constructed as four components

            1. { \lambda_m, Edi_m }_{m=0}^{N}
            2. { \sigma_m, Eri_m }_{m=0}^{N}
            3. { \lambda_m, Edb_m }_{m=0}^{N}
            4. { \sigma_m, Erb_m }_{m=0}^{N}
 
        Energy associated with degenerate eigenmodes are accumulated to a single value. Eigenmodes are deemed
        "degenerate" if their eigenvalues similar out to "decimals" decimal places. The eigenvalue chosen
        for the purpose of the spectra is the average of the eigenvalues of the degenerate modes.
        
        """
        from fluidspectraig.elliptic import vorticity_cgrid as vorticity
        from fluidspectraig.elliptic import divergence_cgrid as divergence

        divu = divergence(u,v,self.splig_n.mask)

        db_m = np.zeros(
            (self.nevals_n), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (boundary)
        di_m = np.zeros(
            (self.nevals_n), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (interior)

        for k in range(0, self.nevals_d):
            ek = self.get_dirichlet_mode(k).data
            di_m[k] = np.sum(divu * ek)  # Projection of divergence onto the neumann modes

            # [TO DO]
            # Need to map the neumann mode from the tracer points to u-points and v-points
            # Then we need to compute int_( div( \vec{u} e_k ) )
            #db_m[k] = -np.sum(divUEta * self.rac)


        vort = vorticity(u,v,self.splig_d.mask)
        vb_m = np.zeros(
            (self.nevals_d), dtype=np.float64
        ) # Projection of vorticity onto the dirichlet modes (boundary)
        vi_m = np.zeros(
            (self.nevals_d), dtype=np.float64
        )  # Projection of vorticity onto the dirichlet modes (interior)

        for k in range(0, self.nevals_d):
            vi_m[k] = np.sum(
                vort * np.squeeze(self.get_dirichlet_mode(k))
            )  # Projection of vorticity onto the dirichlet modes

        # [TO DO]
        Calculate the energy associated with interior vorticity
        Edi = -0.5 * di_m * di_m / self.eval_n
        Edi[self.eval_n == 0.0] = 0.0

        # Calculate the energy associated with boundary vorticity
        Edb = -(0.5 * db_m * db_m + di_m*db_m) / self.eval_n
        Edb[self.eval_n == 0.0] = 0.0

        # Calculate the energy associated with interior vorticity
        Eri = -0.5 * vi_m * vi_m / self.eval_d

        # Calculate the energy associated with boundary vorticity
        Erb = -(0.5 * vb_m * vb_m + vi_m*vb_m) / self.eval_d

        n_evals_rounded = np.round(self.eval_n,decimals=decimals)
        # Collapse degenerate modes
        lambda_m = np.unique(n_evals_rounded)
        Edi_m = np.zeros_like(lambda_m)
        Edb_m = np.zeros_like(lambda_m)
        k = 0
        for ev in lambda_m:
            Edi_m[k] = np.sum(Edi[n_evals_rounded == ev])
            Edb_m[k] = np.sum(Edb[n_evals_rounded == ev])
            k+=1

        d_evals_rounded = np.round(self.eval_d,decimals=decimals) 
        sigma_m = np.unique(d_evals_rounded)
        Eri_m = np.zeros_like(sigma_m)
        Erb_m = np.zeros_like(sigma_m)
        k = 0
        for ev in sigma_m:
            Eri_m[k] = np.sum(Eri[d_evals_rounded == ev])
            Erb_m[k] = np.sum(Erb[d_evals_rounded == ev])
            k+=1

        return lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m

    def plot_eigenmodes(self):
        import matplotlib.pyplot as plt
        import math

        for k in range(int(math.ceil(self.neval_n/6))):
            f,a = plt.subplots(3,2)
            for j in range(6):
                v = self.get_neumann_mode(6*k+j)
                if isinstance(v, np.ndarray):
                    im = a.flatten()[j].imshow(v)
                    f.colorbar(im, ax=a.flatten()[j],fraction=0.046,location='right')
                    a.flatten()[j].set_title(f'e_{6*k+j}')

            plt.tight_layout()
            plt.savefig(f'{self.case_directory}/neumann_modes_{k}.png')
            plt.close()

        for k in range(int(math.ceil(self.neval_d/6))):
            f,a = plt.subplots(3,2)
            for j in range(6):
                v = self.get_dirichlet_mode(6*k+j)
                if isinstance(v, np.ndarray):
                    im = a.flatten()[j].imshow(v)
                    f.colorbar(im, ax=a.flatten()[j],fraction=0.046,location='right')
                    a.flatten()[j].set_title(f'e_{6*k+j}')

            plt.tight_layout()
            plt.savefig(f'{self.case_directory}/dirichlet_modes_{k}.png')
            plt.close()

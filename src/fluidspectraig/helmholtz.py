"""
Helmholtz equation solver with type-I discrete sine transform
and capacitance matrix method.
Louis Thiry, 2023.
"""
# MIT License

# Copyright (c) 2023 louity

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn.functional as F
#from torch_dct import dct_2d, idct_2d

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)


def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs):
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(1,nx, **arr_kwargs),
                          torch.arange(1,ny, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/nx*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/ny*y) - 1)/dy**2


def solve_helmholtz_dst(rhs, helmholtz_dst):
    return F.pad(dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst),
                 (1,1,1,1)
                ).type(torch.float64)


def compute_dst_capacitance_matrices(helmholtz_dst, bound_xids, bound_yids):
    nl  = helmholtz_dst.shape[-3]
    M = bound_xids.shape[0]

    # compute G matrices
    G_matrices = torch.zeros((nl, M, M), dtype=torch.float64, device='cpu')
    rhs = torch.zeros(helmholtz_dst.shape[-3:], dtype=torch.float64,
                      device=helmholtz_dst.device)
    for m in range(M):
        rhs.fill_(0)
        rhs[..., bound_xids[m], bound_yids[m]] = 1
        sol = dstI2D(dstI2D(rhs) / helmholtz_dst.type(torch.float64))
        G_matrices[:,m] = sol[...,bound_xids, bound_yids].cpu()

    # invert G matrices to get capacitance matrices
    capacitance_matrices = torch.zeros_like(G_matrices)
    for l in range(nl):
        capacitance_matrices[l] = torch.linalg.inv(G_matrices[l])

    return capacitance_matrices.to(helmholtz_dst.device)


def solve_helmholtz_dst_cmm(rhs, helmholtz_dst,
                            cap_matrices, bound_xids, bound_yids,
                            mask):
    sol_rect = dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst)
    alphas = torch.einsum(
        '...ij,...j->...i',
        cap_matrices,
        -sol_rect[..., bound_xids, bound_yids].type(torch.float64))
    rhs_2 = rhs.clone()
    rhs_2[..., bound_xids, bound_yids] = alphas
    sol = dstI2D(dstI2D(rhs_2.type(helmholtz_dst.dtype)) / helmholtz_dst).type(torch.float64)
    return F.pad(sol, (1,1,1,1)) * mask


# Discrete cosine transform when neumann boundary conditions are enforced halfway between
# tracer points (on u and v points)
def dctII1D(x, norm='ortho'):
    """1D type-II discrete cosine transform."""
    return torch.fft.irfft(x, dim=-1, norm=norm)[...,0:x.shape[-1]] 

def dctII2D(x, norm='ortho'):
    """2D type-II discrete cosine transform."""
    return dctII1D(dctII1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)


def compute_laplace_dct(nx, ny, dx, dy, arr_kwargs):
    """Discrete cosine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(0,nx, **arr_kwargs),
                          torch.arange(0,ny, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/nx*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/ny*y) - 1)/dy**2

def solve_helmholtz_dct(rhs, helmholtz_dct):
    #b = F.pad(rhs, (1,1,1,1), mode='replicate')
    uhat = dctII2D(rhs)/helmholtz_dct 
    uhat[...,0,0] = 0.0 # enforce zero mean
    return dctII2D(uhat)#[...,1:-1,1:-1]

# def compute_dct_capacitance_matrices(helmholtz_dst, bound_xids, bound_yids):
#     nl  = helmholtz_dst.shape[-3]
#     M = bound_xids.shape[0]

#     # compute G matrices
#     G_matrices = torch.zeros((nl, M, M), dtype=torch.float64, device='cpu')
#     rhs = torch.zeros(helmholtz_dst.shape[-3:], dtype=torch.float64,
#                       device=helmholtz_dst.device)
#     for m in range(M):
#         rhs.fill_(0)
#         rhs[..., bound_xids[m], bound_yids[m]] = 1
#         sol = dstI2D(dstI2D(rhs) / helmholtz_dst.type(torch.float64))
#         G_matrices[:,m] = sol[...,bound_xids, bound_yids].cpu()

#     # invert G matrices to get capacitance matrices
#     capacitance_matrices = torch.zeros_like(G_matrices)
#     for l in range(nl):
#         capacitance_matrices[l] = torch.linalg.inv(G_matrices[l])

#     return capacitance_matrices.to(helmholtz_dst.device)


# def solve_helmholtz_dct_cmm(rhs, helmholtz_dst,
#                             cap_matrices, bound_xids, bound_yids,
#                             mask):
#     sol_rect = dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst)
#     alphas = torch.einsum(
#         '...ij,...j->...i',
#         cap_matrices,
#         -sol_rect[..., bound_xids, bound_yids].type(torch.float64))
#     rhs_2 = rhs.clone()
#     rhs_2[..., bound_xids, bound_yids] = alphas
#     sol = dstI2D(dstI2D(rhs_2.type(helmholtz_dst.dtype)) / helmholtz_dst).type(torch.float64)
#     return F.pad(sol, (1,1,1,1)) * mask

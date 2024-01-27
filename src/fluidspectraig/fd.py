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

import torch.nn.functional as F

def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[...,:-1] - f[...,1:]) / dy, (f[...,1:,:] - f[...,:-1,:]) / dx


def interp_TP(f):
    return 0.25 *(f[...,1:,1:] + f[...,1:,:-1] + f[...,:-1,1:] + f[...,:-1,:-1])


def laplacian_h(f, dx, dy):
    return F.pad(
        (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1]) / dx**2 \
      + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1]) / dy**2,
        (1,1,1,1), mode='constant', value=0.)

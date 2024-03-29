"""Matrix-Free Preconditioned Conjugate Gradient solver with Pytorch

Routines for matrix-free pytorch implementation of the preconditioned 
conjugate gradient method.

"""


import torch

def norm(u):
    """Calculates the magnitude of grid data"""
    return  torch.sqrt( torch.sum(u*u) )

def dot(u,v):
    """Performs dot product on grid data"""
    return torch.sum( u*v )

def pcg(matrixaction, preconditioner, x0, b, tol=1e-12, max_iter=100, arr_kwargs = {'dtype':torch.float64, 'device':'cpu'}):
    """Preconditioned Conjugate Gradient (PCG) Method
    This method applies the Preconditioned Conjugate Gradient Method using a matrix-free
    method for applying the matrix action and the preconditioner.

    Arguments

      matrixaction - A method that takes in data stored in a torch array of x.shape and returns a torch array
                     of shape x.shape. This method is assumed to a linear operator whose matrix-form equivalent
                     is a symmetric positive definite matrix

      preconditioner - A method that takes in data stored in a torch array of x.shape and returns a torch array
                     of shape x.shape. This method is assumed to a linear operator whose matrix-form equivalent
                     is an approximation to the inverse of the matrixaction

      x0            - Initial guess for the solution

      b            - Right hand side for the linear system Ax = b

      tol          - (default: 1e-12) The tolerance for the convergence

      max_iter     - (default: 100) The maximum number of iterations for the IRLM. The IRLM will return when either
                     the max_iter is reached or when the truncation error is less than `tol`, whichever happens first.

      arr_kwargs   - (default: {'dtype':torch.float64, 'device':'cpu'}) Torch object to specify the floating point 
                     precision and the memory locality ('cpu' or 'cuda'). 

    Output
      x  - Torch array of shape x.shape containing the solution to Ax = b
                      
      r - The max-norm residual ||b-Ax||
      
    """

    xk = x0 
    r = b - matrixaction(xk)
    d = preconditioner(r)

    delta = dot(r,d)
    r0 = abs(delta)

    k = 0
    while k < max_iter and abs(delta) > tol*r0:
        q = matrixaction(d)
        alpha = delta / dot(d,q)
        xk += alpha * d

        # update the residual
        r -= alpha * q

        s = preconditioner(r)
        deltaOld = delta
        delta = dot(r,s)
        beta = delta / deltaOld
        d = s + beta * d
        k+=1

    if delta > tol*r0:
        print(
            f"Conjugate gradient method did not converge in {k+1} iterations : {delta}"
        )

    return xk
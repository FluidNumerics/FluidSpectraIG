"""Matrix-Free Krylov solvers with Pytorch

Routines for matrix-free pytorch implementation of 
* preconditioned conjugate gradient method (pcg)
* preconditioned minres (pminres)

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
            f"Conjugate gradient method did not converge in {k+1} iterations : {delta}",
            flush=True
        )

    return xk

def pminres(matrixaction, preconditioner, x0, b, tol=1e-12, max_iter=100, arr_kwargs = {'dtype':torch.float64, 'device':'cpu'}):
    """Preconditioned MINRES 
    This method applies the Preconditioned MINREs Method using a matrix-free
    method for applying the matrix action and the preconditioner.

    See pg. 86 of "Iterative Krylov Methods for Large Linear Systems" - Henk A. van der Vorst

    Arguments

      matrixaction - A method that takes in data stored in a torch array of x.shape and returns a torch array
                     of shape x.shape. This method is assumed to a linear operator whose matrix-form equivalent
                     is a symmetric matrix. The requirement of positive or negative definiteness is relaxed

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
    r = (b - matrixaction(xk)).squeeze()
    v1 = preconditioner( r )
    beta1 = norm(v1)
    beta2 = beta1
    eta = beta1
    gamma1 = 1.0
    gamma0 = 1.0
    sigma1 = 0.0
    sigma0 = 0.0
    v0 = torch.zeros_like(v1)
    w0 = torch.zeros_like(v1)
    wm1 = torch.zeros_like(v1)

    r0 = norm(r)
    dx = r0

    k = 0
    while k < max_iter and abs(dx) > tol*r0:

      # Lanczos iteration
      v1 = v1/beta1
      Av1 = preconditioner(matrixaction(v1)).squeeze()
      alpha1 = dot(v1,Av1)
      v2 = Av1 - alpha1*v1 - beta1*v0
      beta2 = norm(v2)

      # QR Part
      delta = gamma1*alpha1 - gamma0*sigma1*beta1

      rho1 =  torch.sqrt(delta*delta + beta2*beta2)
      rho2 = sigma1*alpha1 + gamma0*gamma1*beta1
      rho3 = sigma0*beta1

      gamma2 = delta/rho1
      sigma2 = beta2/rho1

      w1 = (v1 - rho3*wm1 - rho2*w0)/rho1
      xk += gamma2*eta*w1

      # calculate residual
      dx = torch.abs(gamma2*eta)*norm(w1)

      eta = -sigma2*eta

      gamma0 = gamma1
      gamma1 = gamma2
      wm1 = w0
      w0 = w1
      beta1 = beta2
      sigma0 = sigma1
      sigma1 = sigma2
      v0 = v1
      v1 = v2
      k+=1


    if dx > tol*r0:
        print(
            f"MINRES method did not converge in {k+1} iterations : {dx} ({tol*r0})",
            flush=True
        )

    return xk

def pbicgstab(matrixaction, preconditioner, x0, b, tol=1e-12, max_iter=100, arr_kwargs = {'dtype':torch.float64, 'device':'cpu'}):
    """Preconditioned Bi-Conjugate Gradient Stabilized (Bi-CGStab)
    This method applies the Preconditioned Bi-CGStab Method using a matrix-free
    method for applying the matrix action and the preconditioner.

    See pg. xx of "Iterative Krylov Methods for Large Linear Systems" - Henk A. van der Vorst

    Arguments

      matrixaction - A method that takes in data stored in a torch array of x.shape and returns a torch array
                     of shape x.shape. This method is assumed to a linear operator whose matrix-form equivalent
                     is a symmetric matrix. The requirement of positive or negative definiteness is relaxed

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
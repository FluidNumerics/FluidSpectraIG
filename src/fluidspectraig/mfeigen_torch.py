"""Matrix-Free Eigenmode solver with Pytorch

Routines for matrix-free pytorch implementation of truncated Arnoldi iteration,
truncated Lancsoz iteration, Implicitly Restarted Arnoldi Method, and Implicitly
Restarted Lanczos Method
"""


import torch

def norm(u):
    """Calculates the magnitude of grid data"""
    return  torch.sqrt( torch.sum(u*u) )

def dot(u,v):
    """Performs dot product on grid data"""
    return torch.sum( u*v )


def implicitly_restarted_arnoldi(matrixaction, x, neigs, nkrylov, tol=1e-12, max_iter=100, arr_kwargs = {'dtype':torch.float64, 'device':'cpu'}):

    # Perform the initial m-step arnoldi method
    # V : m arnoldi vectors are in the columns of V
    # H : Hessenburg matrix
    # rmag : remainer magnitude

    k = nkrylov # kyrlov search space size
    m = neigs # number of desired eigenvectors

    H = torch.zeros((k+1, k),**arr_kwargs)
    U = torch.zeros(sum((x.shape, (k+1,)), ()),**arr_kwargs)
    I = torch.eye(k,**arr_kwargs)
    
    # Create an initial guess
    U[...,0] = x/norm(x)

    # Generate the m-step arnoldi factorization
    p = 0
    last_index = arnoldi_iteration(matrixaction, H, U, p, k)
    p = m-1 # Lock the first m columns

    for iter in range(max_iter):
        
        # Find the eigenvalues of H in order
        H_square = H[0:k,0:k]
        H_evals, H_evecs = torch.linalg.eig(H_square) # Eigenvalues are returned in ascending order
        H_evals = H_evals.real
        H_evecs = H_evecs.real
        
        # The residual of the calculation is taken as the norm of the
        # (m+1)-th column vector in the arnoldi iteration.
        rm = torch.abs(H[m,m-1]).cpu().numpy()
        print(f"{iter}, {rm}")
        if rm < tol :
            print(f"IRAM converged in {iter} iterations.")
            # Compute the eigenpairs and return
            ind = torch.argsort(H_evals, descending=False)[0:m]
            eigenvalues = H_evals[ind]
            eigenvectors = U[...,0:k] @ H_evecs[0:k,ind]
            for i in range(m):
                v = eigenvectors[...,i]
                eigenvectors[...,i] = v/norm(v)
            return eigenvalues, eigenvectors, rm

        # Otherwise, we continue our search

        # Implicit shift of the hessenburg (deflate the undesired eigenvalues)
        # First, it is assumed that the matrix associated with the matrix action
        # is negative defininite. Here we sort the eigenvalues in ascending order.
        # We then take the (k-m) largest eigenvalues and use these as "shifts" to 
        # remove the eigenmodes in H-space associated with these eigenvalues.
        ind = torch.argsort(H_evals, descending=False)
        shifts = H_evals[ind]
        Hshift = torch.eye(k,**arr_kwargs) 
        for i in range(m,k):
            work = H_square - H_evals[i]*I
            Hshift = Hshift @ work
        
        # Orthogonalize the columns of Hshift using a QR decomposition
        Q,R = torch.linalg.qr(Hshift,mode='complete') 

        # Update the eigenvectors by projecting onto the deflated basis
        U[...,0:k] = U[...,0:k] @ Q

        # Update the Hessenburg
        H[0:k,0:k] = Q.t() @ H_square @ Q

        # Generate the m+1:k Arnoldi vectors
        last_index = arnoldi_iteration(matrixaction, H, U, p, k)  

    if rm > tol:
        print(f"WARNING: Truncation error estimate not below tolerance : {rm} ({tol})")

    # Compute the eigenpairs and return
    ind = torch.argsort(H_evals, descending=False)[0:m]
    eigenvalues = H_evals[ind]
    eigenvectors = U[...,0:k] @ H_evecs[0:k,ind]
    for i in range(m):
        v = eigenvectors[...,i]
        eigenvectors[...,i] = v/norm(v)
    return eigenvalues, eigenvectors, rm


def arnoldi_iteration(matrixaction, H, U, p, k):
    # This method computes an m-step arnoldi factorization
    #
    # The krylov vectors are set up to retain the
    # shape of x as their shape

    last_index = k

    for j in range(p,k):

        w = matrixaction(U[...,j])

        # Orthogonalize wrt all previous vectors
        for i in range(j+1):
            v = U[...,i]
            H[i,j] = dot(v,w)
            w = w - H[i,j]*v

        # Reorthogonalize to combat round-off error accumulation
        for i in range(j+1):
            v = U[...,i]
            vdotw = dot(v,w)
            w = w - vdotw*v
            H[i,j] = H[i,j] + vdotw

        H[j+1,j] = norm(w)

        if H[j+1,j] < 1e-12:
            print(f"break early {H[j+1,j]}")
            last_index = j
            break

        U[...,j+1] = w/H[j+1,j]
    
    return last_index


    return V, H, rmag


# def implicitly_restarted_arnoldi(matrixaction, x, neigs, nkrylov, tol=1e-6, max_iter=5, max_inner_iter=100, arr_kwargs = {'dtype':torch.float64, 'device':'cpu'}):

#     # Perform the initial m-step arnoldi method
#     # V : m arnoldi vectors are in the columns of V
#     # H : Hessenburg matrix
#     # rmag : remainer magnitude

#     k = nkrylov # kyrlov search space size
#     m = neigs # number of desired eigenvectors

#     H = torch.zeros((k+1, k),**arr_kwargs)
#     U = torch.zeros(sum((x.shape, (k+1,)), ()),**arr_kwargs)
#     I = torch.eye(k,**arr_kwargs)
    
#     # Create an initial guess
#     v0 = x

#     for iter in range(max_iter):

#         # Generate the m-step arnoldi factorization
#         p = 0
#         U[...,0] = v0/norm(v0)
#         last_index = arnoldi_iteration(matrixaction, H, U, p, k)
#         p = m-1 # Lock the first m columns
        
#         for inner_iter in range(max_inner_iter):
#             # Find the eigenvalues of H in order
#             H_square = H[0:k,0:k]
#             H_evals, H_evecs = torch.linalg.eig(H_square) # Eigenvalues are returned in ascending order
#             H_evals = H_evals.real
#             H_evecs = H_evecs.real
            
#             # Implicit shift of the hessenburg (deflate the undesired eigenvalues)
#             # First, it is assumed that the matrix associated with the matrix action
#             # is negative defininite. Here we sort the eigenvalues in descending order according to their absolute value.
#             # We then take the (k-m) largest eigenvalues and use these as "shifts" to 
#             # remove the eigenmodes in H-space associated with these eigenvalues.
#             ind = torch.argsort(torch.abs(H_evals), descending=True)
#             shifts = H_evals[ind]
#             Hshift = torch.eye(k,**arr_kwargs) 
#             for i in range(m,k):
#                 work = H_square - H_evals[i]*I
#                 Hshift = Hshift @ work
            
#             # Orthogonalize the columns of Hshift using a QR decomposition
#             Q,R = torch.linalg.qr(Hshift,mode='complete') 

#             # Update the eigenvectors by projecting onto the deflated basis
#             U[...,0:k] = U[...,0:k] @ Q

#             # Update the Hessenburg
#             H[0:k,0:k] = Q.t() @ H_square @ Q

#             # Generate the m+1:k Arnoldi vectors
#             last_index = arnoldi_iteration(matrixaction, H, U, p, k)
#             print(f"{iter}, {inner_iter}, {H[m,m-1].cpu().numpy()}")
        
#         # Compute the eigenvalues and eigenvectors
#         H_square = H[0:k,0:k]
#         H_evals, H_evecs = torch.linalg.eig(H_square) # Eigenvalues are returned in ascending order
#         H_evals = H_evals.real
#         H_evecs = H_evecs.real
#         ind = torch.argsort(H_evals, descending=False)[0:m]
#         eigenvalues = H_evals[ind]
#         eigenvectors = U[...,0:m] @ H_evecs[0:m,ind]
#         for i in range(m):
#             v = eigenvectors[...,i]
#             eigenvectors[...,i] = v/norm(v)
#         v0 = eigenvectors[...,0]
#         rm = H[m,m-1]
#         if abs(rm) < tol:
#             print(f"IRAM converged in {iter} iterations. Residual = {rm}")
#             break


#     if rm > tol:
#         print(f"WARNING: Truncation error estimate not below tolerance : {rm} ({tol})")

#     return eigenvalues, eigenvectors, rm

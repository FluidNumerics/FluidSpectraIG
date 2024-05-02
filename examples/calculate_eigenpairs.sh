#!/bin/bash
# Notes:
#  * We use the ELPA solver to get all of the eigenpairs
#  * 768x960 case needs > 360 GB memory -- run on GCE instance

CASE_DIR=mitgcm-squarebasin-test/64x64
NP=12

# Compute the dirichlet modes
mpiexec -n ${NP} ../bin/laplacian_modes -f ${CASE_DIR}/dirichlet.dat \
                       -memory_view \
                       -eps_type elpa \
                       -eps_view_vectors hdf5:${CASE_DIR}/dirichlet.evec.h5 \
                       -eps_view_values hdf5:${CASE_DIR}/dirichlet.eval.h5 > ${CASE_DIR}/slepc_dirichlet.stdout

# Compute the neumann modes
mpiexec -n ${NP} ../bin/laplacian_modes -f ${CASE_DIR}/neumann.dat \
                       -memory_view \
                       -eps_type elpa \
                       -eps_view_vectors hdf5:${CASE_DIR}/neumann.evec.h5 \
                       -eps_view_values hdf5:${CASE_DIR}/neumann.eval.h5 > ${CASE_DIR}/slepc_neumann.stdout
/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 2 dimensions.\n\n"
                     "The command line options are:\n"
                     "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
                     "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv)
{
   Mat A;   /* operator matrix */
   EPS eps; /* eigenproblem solver context */
   EPSType type;
   PetscInt N, n, m, nev; // Istart, Iend, II, nev, i, j;
   PetscBool terse;
   char file[PETSC_MAX_PATH_LEN] = ""; /* input file name */
   PetscBool hdf5 = PETSC_FALSE;
   PetscViewer fd; /* viewer */
   char        A_name[128] = "L_d";

   PetscFunctionBeginUser;
   PetscCall(SlepcInitialize(&argc, &argv, (char *)0, help));

   PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), NULL));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n2-D Laplacian Eigenproblem from file = %s\n\n", file ));

   /*
      Decide whether to use the HDF5 reader.
   */
   PetscCall(PetscOptionsGetBool(NULL, NULL, "-hdf5", &hdf5, NULL));

   /*
      Open binary file.  Note that we use FILE_MODE_READ to indicate
      reading from this file.
   */
   if (hdf5)
   {
#if defined(PETSC_HAVE_HDF5)
      PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
      PetscCall(PetscViewerPushFormat(fd, PETSC_VIEWER_HDF5_MAT));
#else
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "PETSc must be configured with HDF5 to use this feature");
#endif
   }
   else
   {
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
   }
   /*
      Load the matrix.
      Matrix type is set automatically but you can override it by MatSetType() prior to MatLoad().
      Do that only if you really insist on the given type.
   */
   PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
   PetscCall(PetscObjectSetName((PetscObject)A, A_name));
   PetscCall(MatSetFromOptions(A));
   PetscCall(MatLoad(A, fd));

   PetscCall(MatGetLocalSize(A, &m, &n));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n2-D Laplacian Eigenproblem, N=%" PetscInt_FMT "\n\n", m));



   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Create the eigensolver and set various options
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   /*
      Create eigensolver context
   */
   PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

   /*
      Set operators. In this case, it is a standard eigenvalue problem
   */
   PetscCall(EPSSetOperators(eps, A, NULL));
   PetscCall(EPSSetProblemType(eps, EPS_HEP));

   /*
      Set solver parameters at runtime
   */
   PetscCall(EPSSetFromOptions(eps));

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve the eigensystem
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   PetscCall(EPSSolve(eps));

   /*
      Optional: Get some information from the solver and display it
   */
   PetscCall(EPSGetType(eps, &type));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Solution method: %s\n\n", type));
   PetscCall(EPSGetDimensions(eps, &nev, NULL, NULL));
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Number of requested eigenvalues: %" PetscInt_FMT "\n", nev));

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Display solution and clean up
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   /* show detailed info unless -terse option is given by user */
   PetscCall(PetscOptionsHasName(NULL, NULL, "-terse", &terse));
   if (terse)
      PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE, NULL));
   else
   {
      PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
      PetscCall(EPSConvergedReasonView(eps, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
   }

   PetscCall(EPSDestroy(&eps));
   PetscCall(MatDestroy(&A));
   PetscCall(SlepcFinalize());
   return 0;
}


hello: hello.o
	-${CLINKER} -o hello hello.o ${SLEPC_SYS_LIB}
	${RM} hello.o

ex1: ex1.o
	-${CLINKER} -o ex1 ex1.o ${SLEPC_EPS_LIB}
	${RM} ex1.o

ex2: ex2.o
	-${CLINKER} -o ex2 ex2.o ${SLEPC_EPS_LIB}
	${RM} ex2.o

laplacian_modes: laplacian_modes.o
	-${CLINKER} -o laplacian_modes laplacian_modes.o ${SLEPC_EPS_LIB}
	${RM} laplacian_modes.o

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
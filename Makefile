
laplacian_modes: src/laplacian_modes.o
	mkdir -p bin/
	-${CLINKER} -o bin/laplacian_modes src/laplacian_modes.o ${SLEPC_EPS_LIB}
	${RM} src/laplacian_modes.o

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
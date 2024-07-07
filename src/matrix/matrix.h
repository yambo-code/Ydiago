#pragma once

#include <stdbool.h>
#include <string.h>
#include "../diago.h"
#include "../SL/scalapack_header.h"
#include <mpi.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

void* BLACScxtInit(char layout, MPI_Comm comm, D_INT ProcX, D_INT ProcY);
void BLACScxtFree(void* mpicontxt);

void* init_D_Matrix(D_INT Grows, D_INT Gcols,
                    D_INT blockX, D_INT blockY, void* mpicontxt);

void free_D_Matrix(void* D_mat);

Err_INT set_descriptor(void* D_mat, D_INT* desc);

Err_INT initiateGetQueue(void* D_mat, const D_LL_INT nelements);

Err_INT DMatGet(void* D_mat, const D_INT i, const D_INT j, D_Cmplx* value);

Err_INT ProcessGetQueue(void* D_mat);

Err_INT initiateSetQueue(void* D_mat, const D_LL_INT nelements);

Err_INT DMatSet(void* D_mat, const D_INT i, const D_INT j, const D_Cmplx value);

Err_INT ProcessSetQueue(void* D_mat);

Err_INT set_identity(void* DmatA); // set it to identity matrix

Err_INT set_zero(void* DmatA); // zero out matrix

Err_INT Construct_BSE_RealHam(void* DmatA, D_float* matA_out);

Err_INT Symplectic_times_L(void* DmatA, D_float* Lmat, D_float* out_Omega_L);

Err_INT Construct_bseW(void* DmatA, D_float* Lmat, D_float* Wmat, char* gpu, void* einfo);

Err_INT BtEig_QLZ(void* DmatA, D_float* Lmat, void* DmatZ, char* gpu, void* einfo);

Err_INT check_mat_diago(void* D_mat, bool even_check);

// elpa related stuff
#ifdef WITH_ELPA
Err_INT set_ELPA(const void* D_mat, const D_INT neigs, const D_INT elpa_solver,
                 const char* gpu_type, const D_INT nthreads,
                 MPI_Comm sub_comm, elpa_t* elpa_handle);

Err_INT cleanup_ELPA(const elpa_t elpa_handle);
#endif

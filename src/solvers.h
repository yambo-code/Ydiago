#pragma once

#include "diago.h"
#include "common/error.h"
// scalapack solvers
// Hermitian solver
Err_INT Heev(void* DmatA, char ulpo, D_INT* neigs_range,
             D_float* eigval_range, D_Cmplx* eig_vals,
             void* Deig_vecs, D_INT* neig_found);

// BSE non-hermtian solver
Err_INT BSE_Solver(void* DmatA, D_INT* neigs_range,
                   D_float* eigval_range, D_Cmplx* eig_vals,
                   void* Deig_vecs, D_INT* neigs_found);

// General non-hermitian solver (unstable)
Err_INT Geev(void* DmatA, D_Cmplx* eig_vals,
             void* Deig_vecsL, void* Deig_vecsR);

// elpa hermitian solver
#ifdef WITH_ELPA
Err_INT Heev_Elpa(void* D_mat, D_Cmplx* eig_vals, void* Deig_vecs,
                  D_INT neigs, const D_INT elpa_solver,
                  const char* gpu_type, const D_INT nthreads);

Err_INT BSE_Solver_Elpa(void* D_mat, D_Cmplx* eig_vals, void* Deig_vecs,
                        const D_INT elpa_solver,
                        char* gpu_type, const D_INT nthreads);
#endif

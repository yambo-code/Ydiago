#pragma once
#include "../diago.h"

#define HAVE_SKEWSYMMETRIC // This is for elpa and must be before elpa/elpa.h
//
#ifdef WITH_ELPA
#include <mpi.h>
#include <stdbool.h>
#include <elpa/elpa.h>
#include "../common/error.h"

#ifdef WITH_DOUBLE
#define Elpa_FunCmplx(FUN_NAME) ElpaCmplx_HIDDEN(FUN_NAME)
#define ElpaCmplx_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_double_complex

#define Elpa_FunFloat(FUN_NAME) ElpaFloat_HIDDEN(FUN_NAME)
#define ElpaFloat_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_double
#else
#define Elpa_FunCmplx(FUN_NAME) ElpaCmplx_HIDDEN(FUN_NAME)
#define ElpaCmplx_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_float_complex

#define Elpa_FunFloat(FUN_NAME) ElpaFloat_HIDDEN(FUN_NAME)
#define ElpaFloat_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_float
#endif

struct ELPAinfo
{
    elpa_t handle;
    MPI_Comm elpa_comm;
    // the MPI communicator. Note we use elpa_comm instead of mat-comm, this
    // is because elpa excepts that all cpus in comm must call the function.
    bool cpu_engage; // True if this function will call the elpa function
};

Err_INT start_ELPA(struct ELPAinfo* info, MPI_Comm comm, const bool cpu_engage);

Err_INT set_ELPA(void* D_mat, const D_INT neigs, const D_INT elpa_solver,
                 const char* gpu_type, const D_INT nthreads, struct ELPAinfo info);

Err_INT cleanup_ELPA(struct ELPAinfo* info);

#endif

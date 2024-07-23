#pragma once

#include "../matrix/matrix.h"
#include "../diago.h"
#include "../solvers.h"
#ifdef WITH_ELPA

#define HAVE_SKEWSYMMETRIC // This is for elpa and must be before elpa/elpa.h
#include <elpa/elpa.h>

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

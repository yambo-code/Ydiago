#pragma once

#include "../matrix/matrix.h"
#include "../diago.h"
#include "../solvers.h"

#ifdef WITH_ELPA

Err_INT set_ELPA(const void* D_mat, const D_INT neigs, const D_INT elpa_solver,
                 const char* gpu_type, const D_INT nthreads,
                 MPI_Comm sub_comm, elpa_t* elpa_handle);

Err_INT cleanup_ELPA(const elpa_t elpa_handle);

#endif

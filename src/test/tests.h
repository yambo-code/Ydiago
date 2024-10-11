#pragma once

#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../SL/scalapack_header.h"
#include "../common/dtypes.h"
#include "../diago.h"
#include "../matrix/matrix.h"
#include "../solvers.h"

#define check_error(a)                               \
    {                                                \
        if ((a))                                     \
        {                                            \
            printf("Error code : %d\n", (int)a);     \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
        }                                            \
    }

#define check_ptr(a)                                 \
    {                                                \
        if (!(a))                                    \
        {                                            \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
        }                                            \
    }

Err_INT load_mat_file(const char* mat_file, const char* eig_file, void* DmatA,
                      D_Cmplx* eig_vals, bool bse_mat);

void copy_mats(void* des_mat, void* src_mat);

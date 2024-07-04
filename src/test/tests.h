#pragma once

#include "../diago.h"
#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
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

Err_INT load_mat_file(const char* mat_file, const char* eig_file, void* DmatA, D_Cmplx* eig_vals, bool bse_mat);

void copy_mats(void* des_mat, void* src_mat);

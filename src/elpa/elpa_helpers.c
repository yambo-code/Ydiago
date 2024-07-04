/* ================== ELPA Helper functions ======================*/

#include "elpa_wrap.h"

#ifdef WITH_ELPA

Err_INT set_ELPA(const void* D_mat, const D_INT neigs, const D_INT elpa_solver,
                 const char* gpu_type, const D_INT nthreads, MPI_Comm sub_comm, elpa_t* elpa_handle)
{
    /*
    Supported types for gputypes : "nvidia-gpu", "amd-gpu", "intel-gpu"
    Value are only set when they are > 0, if <0, they will be ignored
    */
    int error = check_mat_diago(D_mat, false);
    if (error)
    {
        return error;
    }
    const struct D_Matrix* mat = D_mat;

    if (elpa_init(SUPPORTED_ELPA_VERSION) != ELPA_OK)
    { // elpa version is not supported
        return ELPA_UNSUPPORTED_ERROR;
    }

    elpa_t handle = elpa_allocate(&error);
    *elpa_handle = handle;

    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    /* Set parameters the matrix and it's MPI distribution */
    elpa_set_integer(handle, "na", mat->gdims[0], &error);
    // Global dim of matrix
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    if (neigs > 0)
    {
        D_INT neigs_tmp = MIN(neigs, mat->gdims[0]);
        if (!neigs)
        {
            neigs_tmp = mat->gdims[0];
        }
        elpa_set_integer(handle, "nev", neigs_tmp, &error);
        // number of eigenvectors requested
        if (error != ELPA_OK)
        {
            return ELPA_SETUP_ERROR;
        }
    }

    elpa_set_integer(handle, "local_nrows", mat->ldims[0], &error);
    // number of local rows of the distributed matrix on this MPI task
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    elpa_set_integer(handle, "local_ncols", mat->ldims[1], &error);
    // number of local columns of the distributed matrix on this MPI task
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    elpa_set_integer(handle, "nblk", mat->block_size[0], &error);
    // size of the BLACS block cyclic distribution
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(sub_comm), &error);
    // the MPI communicator. Note we use sub_comm instead of mat-comm, this
    // is because elpa excepts that all cpus in comm call the function.
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    elpa_set_integer(handle, "process_row", mat->pids[0], &error);
    // row coordinate of MPI process
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    elpa_set_integer(handle, "process_col", mat->pids[1], &error);
    // column coordinate of MPI process
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    // setup the handle
    error = elpa_setup(handle);
    if (error != ELPA_OK)
    {
        return ELPA_SETUP_ERROR;
    }

    // set openmp
#ifdef WITH_OPENMP
    // elpa openmp only works when no gpu is present
    if (nthreads > 0)
    {
        elpa_set_integer(handle, "omp_threads", nthreads, &error);
        if (error != ELPA_OK)
        {
            return ELPA_SETUP_ERROR;
        }
    }
#endif

    // set gpus
#ifdef WITH_GPU
    if (gpu_type)
    {
        elpa_set_integer(handle, gpu_type, 1, &error);
        if (error != ELPA_OK)
        {
            return ELPA_SETUP_ERROR;
        }

        error = elpa_setup_gpu(handle);
        if (error != ELPA_OK)
        {
            return ELPA_SETUP_ERROR;
        }
    }
#endif

    if (elpa_solver > 0)
    {
        // set the type of solver
        if (elpa_solver == 1)
        {
            elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
            if (error != ELPA_OK)
            {
                return ELPA_SETUP_ERROR;
            }
        }
        else
        {
            elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
            if (error != ELPA_OK)
            {
                return ELPA_SETUP_ERROR;
            }
        }
    }
    return error;
}

Err_INT cleanup_ELPA(const elpa_t elpa_handle)
{
    //  cleanup
    int error;
    elpa_deallocate(elpa_handle, &error);
    if (error != ELPA_OK)
    {
        return ELPA_DEALLOC_ERROR;
    }

    elpa_uninit(&error);
    if (error != ELPA_OK)
    {
        return ELPA_UNINIT_ERROR;
    }

    return error;
}

#endif
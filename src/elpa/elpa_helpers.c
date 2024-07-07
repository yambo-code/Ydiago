/* ================== ELPA Helper functions ======================*/

#include "elpa_wrap.h"

#ifdef WITH_ELPA

Err_INT start_ELPA(struct ELPAinfo* info, MPI_Comm comm, const bool cpu_engage)
{
    if (!info)
    {
        return ERR_NULL_PTR_BUFFER;
    }

    info->cpu_engage = cpu_engage;

    int colour_diago_comm = 0;

    if (!cpu_engage)
    {
        colour_diago_comm = 1;
    }

    int my_rank_comm;
    if (MPI_Comm_rank(comm, &my_rank_comm))
    {
        return DIAGO_MPI_ERROR;
    }

    if (MPI_Comm_split(comm, colour_diago_comm, my_rank_comm, &info->elpa_comm))
    {
        return DIAGO_MPI_ERROR;
    }

    if (cpu_engage)
    {
        if (elpa_init(SUPPORTED_ELPA_VERSION) != ELPA_OK)
        { // elpa version is not supported
            return ELPA_UNSUPPORTED_ERROR;
        }

        int error = 0;
        info->handle = elpa_allocate(&error);

        if (error != ELPA_OK)
        {
            return ELPA_SETUP_ERROR;
        }
    }

    return DIAGO_SUCCESS;
}

Err_INT set_ELPA(const void* D_mat, const D_INT neigs, const D_INT elpa_solver,
                 const char* gpu_type, const D_INT nthreads, struct ELPAinfo info)
{
    /*
    Supported types for gputypes : "nvidia-gpu", "amd-gpu", "intel-gpu"
    Value are only set when they are > 0, if <0, they will be ignored
    */
    if (!info.cpu_engage)
    {
        return DIAGO_SUCCESS; // cpu does not participate
    }

    MPI_Comm elpa_comm = info.comm;
    elpa_t handle = info.handle;

    int error = check_mat_diago(D_mat, false);
    if (error)
    {
        return error;
    }
    const struct D_Matrix* mat = D_mat;

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

    elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(elpa_comm), &error);

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

        elpa_set_integer(handle, "gpu_hermitian_multiply", 1, &error);
        if (error != ELPA_OK)
        {
            return ELPA_SETUP_ERROR;
        }

        elpa_set_integer(handle, "gpu_cholesky", 1, &error);
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

Err_INT cleanup_ELPA(struct ELPAinfo* info)
{
    if (!info)
    {
        return DIAGO_SUCCESS;
    }

    //  cleanup
    int error = 0, error2 = 0;

    if (info->cpu_engage)
    {

        elpa_deallocate(info->handle, &error);
        if (error != ELPA_OK)
        {
            error = ELPA_DEALLOC_ERROR;
        }

        elpa_uninit(&error2);
        if (!error && error2 != ELPA_OK)
        {
            error = ELPA_UNINIT_ERROR;
        }
    }

    error2 = MPI_Comm_free(&info->elpa_comm);
    // Bcast eigen values to all cpus
    if (!error && error2)
    {
        error = DIAGO_MPI_ERROR;
    }

    return error;
}

#endif
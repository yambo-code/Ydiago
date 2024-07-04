#include "elpa_wrap.h"

#ifdef WITH_ELPA

// FIX MEEEE USe type specific elpa interface

Err_INT Heev_Elpa(const void* D_mat, D_Cmplx* eig_vals, void* Deig_vecs,
                  D_INT neigs, const D_INT elpa_solver,
                  const char* gpu_type, const D_INT nthreads)
{
    // NOTE : full matrix must be filled for ELPA.

    if (!eig_vals)
    {
        return ERR_NULL_PTR_BUFFER;
        // This is a fatal error return immediately
    }

    int error = check_mat_diago(D_mat, false);

    if (error)
    {
        goto elpa_herm_end0;
    }

    const struct D_Matrix* mat = D_mat;
    struct D_Matrix* evecs = Deig_vecs;

    if (neigs < 1 || neigs > mat->gdims[0])
    {
        // errornous number of eigen values. compute all of them
        neigs = mat->gdims[0];
    }

    // we need to create a comm that participates in diagonalization
    MPI_Comm diago_comm;
    int colour_diago_comm = 0;

    if (!mat->cpu_engage)
    {
        colour_diago_comm = 1;
    }

    int my_rank_comm;
    if (MPI_Comm_rank(mat->comm, &my_rank_comm))
    {
        error = DIAGO_MPI_ERROR;
        goto elpa_herm_end;
    }

    if (MPI_Comm_split(mat->comm, colour_diago_comm, my_rank_comm, &diago_comm))
    {
        error = DIAGO_MPI_ERROR;
        goto elpa_herm_end;
    }

    if (colour_diago_comm)
    {
        goto elpa_herm_end;
    }

    D_float* evals = calloc(mat->gdims[0], sizeof(*evals));
    if (evals)
    {
        elpa_t elpa_handle;
        error = set_ELPA(D_mat, neigs, elpa_solver, gpu_type, nthreads, diago_comm, &elpa_handle);
        if (!error)
        {
            if (evecs)
            {
                elpa_eigenvectors(elpa_handle, mat->data, evals, evecs->data, &error);
            }
            else
            {
                elpa_eigenvalues(elpa_handle, mat->data, evals, &error);
            }
        }
        if (!error)
        {
            for (D_LL_INT i = 0; i < neigs; ++i)
            {
                eig_vals[i] = evals[i];
            }
        }
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }

    free(evals);
    error = cleanup_ELPA(elpa_handle);

elpa_herm_end:;
    // free the new comm
    mpi_error = MPI_Comm_free(&diago_comm);
    // Bcast eigen values to all cpus
elpa_herm_end0:;
    mpi_error = MPI_Bcast(eig_vals, neigs, D_Cmplx_MPI_TYPE, 0, mat->comm);

    return error;
}

#endif
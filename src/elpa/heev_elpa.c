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
        return error; // fatal error
    }

    const struct D_Matrix* mat = D_mat;
    struct D_Matrix* evecs = Deig_vecs;

    if (neigs < 1 || neigs > mat->gdims[0])
    {
        // errornous number of eigen values. compute all of them
        neigs = mat->gdims[0];
    }

    // initiate values of eigvals to be 0
    for (D_LL_INT i = 0; i < mat->gdims[0]; ++i)
    {
        eig_vals[i] = 0;
    }

    struct ELPAinfo Einfo;
    // collective call
    error = start_ELPA(&Einfo, mat->comm, mat->cpu_engage);
    if (error)
    {
        goto elpa_herm_end;
    }

    if (!mat->cpu_engage)
    {
        goto elpa_herm_end;
    }

    D_float* evals = calloc(mat->gdims[0], sizeof(*evals));
    if (evals)
    {
        error = set_ELPA(D_mat, neigs, elpa_solver, gpu_type, nthreads, Einfo);
        if (!error)
        {
            if (evecs)
            {
                Elpa_FunCmplx(eigenvectors)(Einfo.handle, mat->data, evals, evecs->data, &error);
            }
            else
            {
                Elpa_FunCmplx(eigenvalues)(Einfo.handle, mat->data, evals, &error);
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

elpa_herm_end:;

    int error1 = cleanup_ELPA(&Einfo);
    if (!error)
    {
        error = error1;
    }

    mpi_error = MPI_Bcast(eig_vals, neigs, D_Cmplx_MPI_TYPE, 0, mat->comm);
    if (!error && mpi_error)
    {
        error = DIAGO_MPI_ERROR;
    }

    return error;
}

#endif
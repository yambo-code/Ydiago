#ifdef WITH_ELPA
//
#include "elpa_wrap.h"
#include "../diago.h"
#include <mpi.h>
#include "../common/error.h"
#include "../common/dtypes.h"
#include <elpa/elpa.h>
#include "../matrix/matrix.h"
#include "../common/min_max.h"
#include "../common/gpu_helpers.h"
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "../solvers.h"


static void elpa_eig_vals_gpu(elpa_t handle, D_Cmplx* a, D_LL_INT a_nele, D_float* ev,
                              D_LL_INT ev_nele, int* error);

static void elpa_eig_vecs_gpu(elpa_t handle, D_Cmplx* a, D_LL_INT a_nele, D_float* ev,
                              D_LL_INT ev_nele, D_Cmplx* q, D_LL_INT q_nele, int* error);

Err_INT Heev_Elpa(void* D_mat, D_Cmplx* eig_vals, void* Deig_vecs,
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
    //
    bool gpu_calc = isGPUpresent() && gpu_type;
    //
    if (evals)
    {
        error = set_ELPA(D_mat, neigs, elpa_solver, gpu_type, nthreads, Einfo);
        if (!error)
        {
            if (evecs)
            {
                if (!gpu_calc)
                {
                    elpa_eigenvectors(Einfo.handle, mat->data, evals, evecs->data, &error);
                }
                else
                {
                    elpa_eig_vecs_gpu(Einfo.handle, mat->data, mat->ldims[0] * mat->ldims[1], evals,
                                      mat->gdims[0], evecs->data, evecs->ldims[0] * evecs->ldims[1], &error);
                }
            }
            else
            {
                if (!gpu_calc)
                {
                    elpa_eigenvalues(Einfo.handle, mat->data, evals, &error);
                }
                else
                {
                    elpa_eig_vals_gpu(Einfo.handle, mat->data, mat->ldims[0] * mat->ldims[1], evals,
                                      mat->gdims[0], &error);
                }
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

    int mpi_error = MPI_Bcast(eig_vals, neigs, D_Cmplx_MPI_TYPE, 0, mat->comm);
    if (!error && mpi_error)
    {
        error = DIAGO_MPI_ERROR;
    }

    return error;
}

static void elpa_eig_vecs_gpu(elpa_t handle, D_Cmplx* a, D_LL_INT a_nele, D_float* ev,
                              D_LL_INT ev_nele, D_Cmplx* q, D_LL_INT q_nele, int* error)
{
    *error = 0;
    if (!isGPUpresent())
    {
        return;
    }
#ifdef WITH_GPU
    D_Cmplx* a_dev = gpu_malloc((a_nele + 1) * sizeof(*a_dev));
    D_float* ev_dev = gpu_malloc((ev_nele + 1) * sizeof(*ev_dev));
    D_Cmplx* q_dev = gpu_malloc((q_nele + 1) * sizeof(*q_dev));

    if (a_dev)
    {
        *error = gpu_memcpy(a_dev, a, a_nele * sizeof(*a_dev), Copy2GPU);
    }

    if (!*error && a_dev && ev_dev && q_dev)
    {
        Elpa_FunCmplx(eigenvectors)(handle, a_dev, ev_dev, q_dev, error);

        if (!*error)
        {
            // Copy back to cpu
            *error = gpu_memcpy(ev, ev_dev, ev_nele * sizeof(*ev), Copy2CPU);
            *error = gpu_memcpy(q, q_dev, q_nele * sizeof(*q), Copy2CPU) || *error;
        }
    }
    else
    {
        *error = 1;
    }

    *error = gpu_free(a_dev) || *error;
    *error = gpu_free(ev_dev) || *error;
    *error = gpu_free(q_dev) || *error;
#else
    return;
#endif
}

static void elpa_eig_vals_gpu(elpa_t handle, D_Cmplx* a, D_LL_INT a_nele, D_float* ev,
                              D_LL_INT ev_nele, int* error)
{
    *error = 0;
    if (!isGPUpresent())
    {
        return;
    }
#ifdef WITH_GPU
    D_Cmplx* a_dev = gpu_malloc((a_nele + 1) * sizeof(*a_dev));
    D_float* ev_dev = gpu_malloc((ev_nele + 1) * sizeof(*ev_dev));
    if (a_dev)
    {
        *error = gpu_memcpy(a_dev, a, a_nele * sizeof(*a_dev), Copy2GPU);
    }

    if (!*error && a_dev && ev_dev)
    {
        Elpa_FunCmplx(eigenvalues)(handle, a_dev, ev_dev, error);

        if (!*error)
        {
            // Copy back to cpu
            *error = gpu_memcpy(ev, ev_dev, ev_nele * sizeof(*ev), Copy2CPU);
        }
    }
    else
    {
        *error = 1;
    }

    *error = gpu_free(a_dev) || *error;
    *error = gpu_free(ev_dev) || *error;
#else
    return;
#endif
}

#endif

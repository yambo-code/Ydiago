// Contain Special non-TDA BSE Solver
#include "../solvers.h"
#include "../matrix/matrix.h"
#include "../SL/scalapack_header.h"
#include "elpa_wrap.h"

#ifdef WITH_ELPA

Err_INT BSE_Solver_Elpa(void* D_mat, D_Cmplx* eig_vals, void* Deig_vecs,
                        const D_INT elpa_solver,
                        char* gpu_type, const D_INT nthreads)
{
    /*
    Note the eig_vals buffer must have the dimension of the matrix
    */
    /*
    The Matrix must have the the following structure
    |  A    B   |
    | -B*  -A*  |
    Where A is hermitian and B is symmetric
    And moreover.
    |  A    B   |
    |  B*   A*  | must be postive definite.
    */
    // This method is based on
    // https://doi.org/10.1016/j.laa.2015.09.036

    /*

    DmatA is destroyed

    Eigen values come in pair i.e (-lambda, lambda).
    Only computes postive eigenvalues and their correspoing right
    eigen vectors. From this we can retreive left eigen vectors
    for positive eigen values, and left and right eigenvectors of negative
    eigen values

        Right eigenvectors         Left eigenvectors
          +ve     -ve               +ve       -ve
    X = [ X_1, conj(X_2) ]    Y = [  X_1, -conj(X_2)]
*        [X_2, conj(X_1) ]        [ -X_2,  conj(X_1)],

    The code returns [X1,X2] as eigen vectors
    */

    int err_code = 0;

    Err_INT error = check_mat_diago(DmatA, true);
    if (error)
    {
        goto end_BSE_Solver0;
    }

    struct D_Matrix* matA = DmatA;
    struct D_Matrix* matZ = Deig_vecs;

    if (!eig_vals)
    {
        return ERR_NULL_PTR_BUFFER;
        // This is a fatal error, better return immediately!
    }
    else
    {
        // zero out the eigenvalue buffer
        for (D_LL_INT i = 0; i < matA->gdims[0]; ++i)
        {
            eig_vals[i] = 0;
        }
    }

    D_INT ndim = matA->gdims[0] / 2;

    D_INT neigs = matA->gdims[0] / 2;

    D_LL_INT nloc_elem = matA->ldims[0] * matA->ldims[1];

    D_float* Ham_r = calloc(nloc_elem + 1, sizeof(*Ham_r));
    if (!Ham_r)
    {
        error = BUF_ALLOC_FAILED;
        goto end_BSE_Solver0;
    }

    D_INT izero = 1;

    D_INT desca[9];
    error = set_descriptor(matA, desca);
    if (error)
    {
        goto end_BSE_Solver1;
    }

    // 1) compute the real hamilitian
    error = Construct_BSE_RealHam(matA, Ham_r);

    if (error)
    {
        goto end_BSE_Solver1;
    }

    // Elpa setup
    struct ELPAinfo einfo;
    error = start_ELPA(&einfo, matA->comm, matA->cpu_engage);
    if (error)
    {
        goto end_BSE_Solver1;
    }

    error = set_ELPA(D_mat, neigs, elpa_solver, gpu_type, nthreads, &einfo);
    if (error)
    {
        goto end_BSE_Solver1;
    }

    // 2) Perform the Cholesky factorization for real symmetric matrix
    if (matA->cpu_engage)
    {
        // FIX ME : Port it to GPU
        Elpa_FunFloat(cholesky)(einfo.handle, Ham_r, &err_code)
    }
    // L is stotred in Ham
    if (err_code)
    {
        error = CHOLESKY_FAILED;
        goto end_BSE_Solver2;
    }

    // 3) W = L^T \omega * L
    /*
    Note that this is real. and W is skew symmetric.
    */
    D_float* Wmat = calloc(nloc_elem + 1, sizeof(*Wmat));

    if (Wmat)
    {

        error = Construct_bseW(matA, Ham_r, Wmat, gpu_type, &einfo);
        /*
        For Elpa there is real skew symmetric solver, so we use it,
        We need postive eigen values of (-iW), where W is skew symmetric
        As elpa only computes the first n eigen values, we compute the first n negative (ofcourse imaginary)
        eigen values of skew symmetric matrix (-W), which correspond to positve eigenvalues of
        -iW. Due to this, we always are obliged to request the full spectrum with ELPA
        */
        if (!error)
        {
            for (D_LL_INT i = 0; i < nloc_elem; ++i)
            {
                Wmat[i] = -Wmat[i];
            }
        }
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }

    if (!error && matA->cpu_engage)
    {
        //(elpa_t handle, double *a, double *ev, double *q, int *error)
        D_float* evals_tmp = calloc(matA->gdims[0], sizeof(*evals_tmp));
        D_float* evecs_tmp = calloc(2 * nloc_elem + 1, sizeof(*evecs_tmp));

        if (evals_tmp && evecs_tmp)
        {
            // Port it to GPU
            Elpa_FunFloat(skew_eigenvectors)(einfo.handle, Wmat, evals_tmp, evecs_tmp, &err_code);
            if (err_code != ELPA_OK)
            {
                error = ELPA_SKEW_SYMM_DIAGO_ERROR;
            }
            else
            {
                for (D_LL_INT i = 0; i < neigs; ++i)
                {
                    // -ve sign because we diagonalized -W instead of W
                    evals_tmp[i] = -evals_tmp[i];
                    eig_vals[i] = evals_tmp[i];
                }

                D_INT error_sl = 0;

                D_INT lwork = MAX(matA->gdims[0], (mat->ldims[0] * (matA->block_size[1] + mat->ldims[1])));
                D_INT liwork = matA->gdims[0] + 2 * matA->block_size[1] + 2 * (MAX(matA->pgrid[0], matA->pgrid[1]));

                D_float* work = calloc(lwork, sizeof(*work));
                D_INT* iwork = calloc(liwork, sizeof(*iwork));
                if (work && iwork)
                {
                    // sort real
                    SL_FunFloat(lasrt)("I", &neigs, evals_tmp, evecs_tmp,
                                       &izero, &izero, desca, work,
                                       &lwork, iwork, &liwork, &error_sl);

                    for (D_LL_INT i = 0; i < neigs; ++i)
                    {
                        evals_tmp[i] = creal(eig_vals[i]);
                    }

                    if (error_sl)
                    {
                        error = SL_EIG_SORT_ERROR;
                    }
                    // sort imag
                    SL_FunFloat(lasrt)("I", &neigs, evals_tmp, evecs_tmp + nloc_elem,
                                       &izero, &izero, desca, work,
                                       &lwork, iwork, &liwork, &error_sl);

                    if (!error && error_sl)
                    {
                        error = SL_EIG_SORT_ERROR;
                    }

                    for (D_LL_INT i = 0; i < neigs; ++i)
                    {
                        eig_vals[i] = evals_tmp[i];
                    }
                }
                else
                {
                    error = SL_WORK_SPACE_ERROR;
                }
                free(work);
                free(iwork);

                // Copy the eigen vectors to matA
                for (D_LL_INT i = 0; i < nloc_elem; ++i)
                {
                    matZ->data[i] = evecs_tmp[i] + I * evecs_tmp[i + nloc_elem];
                }
            }
        }
        else
        {
            error = BUF_ALLOC_FAILED;
        }
        free(evecs_tmp);
        free(evals_tmp);
    }

    free(Wmat);

    if (error)
    {
        goto end_BSE_Solver2;
    }

    if (Deig_vecs)
    {
        // FIX ME : check eige_vecs

        // back transform eigen vectors
        // [X1, X2] = [[I_n , 0] [0, -I_n]] @ Q @ L @ Z@\lambda^**-1/2
        error = BtEig_QLZ(matA, Ham_r, Deig_vecs, gpu_type);
        if (error)
        {
            goto end_BSE_Solver2;
        }

        D_INT descz[9];
        error = set_descriptor(Deig_vecs, descz);
        if (error)
        {
            goto end_BSE_Solver2;
        }

        if (matA->cpu_engage)
        {
            // normalize the eigen vectors with |2*lambda|**-0.5
            for (D_INT i = 0; i < neigs; ++i)
            {
                D_Cmplx alpha = csqrt(2 * cabs(eig_vals[i]));

                if (cabs(alpha) < 1e-8)
                {
                    // FIX ME : NM: This is an error because we only want eigen values >0
                    continue;
                }
                else
                {
                    alpha = 1.0 / alpha;
                }
                D_INT jx = i + 1;

                SL_FunCmplx(scal)(matZ->gdims, &alpha, matZ->data, &izero, &jx, descz, &izero);
            }
            // sort the eigen values the eigen vectors according to the absoulte value
            // FIX ME : Sort
        }
    }

end_BSE_Solver2:;
    cleanup_ELPA(&einfo);

end_BSE_Solver1:;
    free(Ham_r);
    // Bcast eigen values to all cpus
end_BSE_Solver0:;
    return error;
}
#endif
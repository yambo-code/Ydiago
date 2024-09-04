// Contain Special non-TDA BSE Solver
#include "../solvers.h"
#include "../matrix/matrix.h"
#include "../SL/scalapack_header.h"
#include "elpa_wrap.h"

#ifdef WITH_ELPA

static void elpa_skew_eig_vecs_gpu(elpa_t handle, D_float* a, D_LL_INT a_nele, D_float* ev,
                                   D_LL_INT ev_nele, D_float* q, D_LL_INT q_nele, int* error);

static void elpa_cholesky_real_gpu(elpa_t handle, D_float* a, D_LL_INT a_nele, int* error);

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

    D_mat is destroyed

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
    bool gpu_calc = isGPUpresent() && gpu_type;

    Err_INT error = check_mat_diago(D_mat, true);
    if (error)
    {
        goto end_BSE_Solver0;
    }

    struct D_Matrix* matA = D_mat;
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

    error = set_ELPA(D_mat, neigs, elpa_solver, gpu_type, nthreads, einfo);
    if (error)
    {
        goto end_BSE_Solver1;
    }

    D_float* Wmat = calloc(nloc_elem + 1, sizeof(*Wmat)); // W = L^T \omega * L

    if (Wmat)
    {
        // 2) Perform the Cholesky factorization for real symmetric matrix
        if (matA->cpu_engage)
        {
            if (!gpu_calc)
            {
                elpa_cholesky(einfo.handle, Ham_r, &err_code);
            }
            else
            {
                elpa_cholesky_real_gpu(einfo.handle, Ham_r, nloc_elem, &err_code);
            }
            //  Elpa gives upper triangular. So transpose
            D_float alpha_tmp = 1.0;
            D_float beta_tmp = 0.0;
            SL_FunFloat(tradd)("L", "T", matA->gdims, matA->gdims + 1,
                               &alpha_tmp, Ham_r, &izero, &izero,
                               desca, &beta_tmp, Wmat, &izero, &izero, desca);

            // swap the Ham_r and Wmat pointer to avoid copying
            D_float* tmp_ptr = Wmat;
            Wmat = Ham_r;
            Ham_r = tmp_ptr;
        }

        // L is stotred in Ham
        if (err_code)
        {
            error = CHOLESKY_FAILED;
        }
        else
        {

            // 3) COmpute W = L^T \omega * L
            /*
            Note that this is real. and W is skew symmetric.
            */

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
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }

    if (!error && matA->cpu_engage)
    {
        D_float* evals_tmp = calloc(matA->gdims[0], sizeof(*evals_tmp));
        D_float* evecs_tmp = calloc(2 * nloc_elem + 1, sizeof(*evecs_tmp));

        if (evals_tmp && evecs_tmp)
        {
            if (!gpu_calc)
            {
                elpa_skew_eigenvectors(einfo.handle, Wmat, evals_tmp, evecs_tmp, &err_code);
            }
            else
            {
                elpa_skew_eig_vecs_gpu(einfo.handle, Wmat, nloc_elem, evals_tmp,
                                       matA->gdims[0], evecs_tmp, 2 * nloc_elem, &err_code);
            }
            if (err_code)
            {
                error = ELPA_SKEW_SYMM_DIAGO_ERROR;
            }
            else
            {
                // the first eigen value is the least one i.e with maximum absoulte value
                D_float max_eig_value = fabs(evals_tmp[0]);
                for (D_LL_INT i = 0; i < neigs; ++i)
                {
                    // -ve sign because we diagonalized -W instead of W
                    evals_tmp[i] = -evals_tmp[i];
                    eig_vals[i] = evals_tmp[i];
                }
                for (D_LL_INT i = neigs; i < matA->gdims[0]; ++i)
                {
                    // fill the rest with are dummy values in accending order.
                    evals_tmp[i] = max_eig_value + i + 1.0;
                }
                // The skew symmetric eigen vectors are arranged as [:,:eigs,:2].

                D_INT error_sl = 0;

                D_INT lwork = MAX(matA->gdims[0], (matA->ldims[0] * (matA->block_size[1] + matA->ldims[1])));
                // NM : liwork is intentionally kept more than what is given in sl to avoid buffer overflow (sl requirment is wrong!)
                // From the p?lasrt routine, we need at lease 2*N
                D_INT liwork = 2*matA->gdims[0] + 2 * matA->block_size[1] + 2 * (MAX(matA->pgrid[0], matA->pgrid[1]));

                D_float* work = calloc(lwork, sizeof(*work));
                D_INT* iwork = calloc(liwork, sizeof(*iwork));
                if (work && iwork)
                {
                    // sort real
                    SL_FunFloat(lasrt)("I", matA->gdims, evals_tmp, evecs_tmp,
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
                    SL_FunFloat(lasrt)("I", matA->gdims, evals_tmp, evecs_tmp + nloc_elem,
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
        error = BtEig_QLZ(matA, Ham_r, Deig_vecs, neigs, gpu_type, &einfo);
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

static void elpa_cholesky_real_gpu(elpa_t handle, D_float* a, D_LL_INT a_nele, int* error)
{
    *error = 0;
    if (!isGPUpresent())
    {
        return;
    }
#ifdef WITH_GPU
    D_float* a_dev = gpu_malloc((a_nele + 1) * sizeof(*a_dev));

    if (a_dev)
    {
        *error = gpu_memcpy(a_dev, a, a_nele * sizeof(*a_dev), Copy2GPU);
    }

    if (!*error && a_dev)
    {
        Elpa_FunFloat(cholesky)(handle, a_dev, error);
        if (!*error)
        {
            // Copy back to cpu
            *error = gpu_memcpy(a, a_dev, a_nele * sizeof(*a_dev), Copy2CPU);
        }
    }
    else
    {
        *error = 1;
    }

    *error = gpu_free(a_dev) || *error;
#else
    return;
#endif
}

static void elpa_skew_eig_vecs_gpu(elpa_t handle, D_float* a, D_LL_INT a_nele, D_float* ev,
                                   D_LL_INT ev_nele, D_float* q, D_LL_INT q_nele, int* error)
{
    *error = 0;
    if (!isGPUpresent())
    {
        return;
    }
#ifdef WITH_GPU
    D_float* a_dev = gpu_malloc((a_nele + 1) * sizeof(*a_dev));
    D_float* ev_dev = gpu_malloc((ev_nele + 1) * sizeof(*ev_dev));
    D_float* q_dev = gpu_malloc((q_nele + 1) * sizeof(*q_dev));

    if (a_dev)
    {
        *error = gpu_memcpy(a_dev, a, a_nele * sizeof(*a_dev), Copy2GPU);
    }

    if (!*error && a_dev && ev_dev && q_dev)
    {
#ifdef WITH_DOUBLE
        elpa_skew_eigenvectors_d_ptr_d(handle, a_dev, ev_dev, q_dev, error);
#else
        elpa_skew_eigenvectors_d_ptr_f(handle, a_dev, ev_dev, q_dev, error);
#endif // with_double
       //
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

#endif // WITH_ELPA

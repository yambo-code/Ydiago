// Contain Special non-TDA BSE Solver
#include "../solvers.h"
#include "../matrix/matrix.h"
#include "scalapack_header.h"

Err_INT BSE_Solver(void* DmatA, D_INT* neigs_range,
                   D_float* eigval_range, D_Cmplx* eig_vals,
                   void* Deig_vecs, D_INT* neigs_found)
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

    D_INT err_code = 0;

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

    // 2) Perform the Cholesky factorization for real symmetric matrix
    if (matA->cpu_engage)
    {
        SL_FunFloat(potrf)("L", matA->gdims, Ham_r, &izero, &izero, desca, &err_code);
    }
    // L is stotred in Ham
    if (err_code)
    {
        error = CHOLESKY_FAILED;
        goto end_BSE_Solver1;
    }

    // 3) W = L^T \omega * L
    /*
    Note that this is real. and W is skew symmetric.
    */
    D_float* Wmat = calloc(nloc_elem + 1, sizeof(*Wmat));

    if (Wmat)
    {

        error = Construct_bseW(matA, Ham_r, Wmat, NULL, NULL);

        /*
        Since there are no solvers for skew symmetric in scalapack,
        so we use the complex hermitain
        solvers to diagonalize -iW.
        */

        // compute -iW and diagonalize using hermitian solver (only positive eigen values)
        if (!error)
        {
            for (D_LL_INT i = 0; i < nloc_elem; ++i)
            {
                matA->data[i] = -I * Wmat[i];
            }
        }
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }
    free(Wmat);

    if (error)
    {
        goto end_BSE_Solver1;
    }

    // first set the default
    D_INT ev_tmp[2] = { ndim + 1, 2 * ndim };
    D_INT* n_ev_range = ev_tmp;
    // compute the eigen vectors
    if (neigs_range)
    {
        // neig_range should start from [ndim+1]
        D_INT neigs_min = MIN(neigs_range[0], neigs_range[1]);
        D_INT neigs_max = MAX(neigs_range[0], neigs_range[1]);
        if (neigs_min <= ndim)
        {
            neigs_min = ndim + 1;
        }
        n_ev_range[0] = neigs_min;
        n_ev_range[1] = neigs_max;
    }
    if (eigval_range)
    {
        // eigvals should start from 0.0
        D_float eigval_min = MIN(eigval_range[0], eigval_range[1]);
        D_float eigval_max = MAX(eigval_range[0], eigval_range[1]);
        if (eigval_min < 0)
        {
            eigval_min = 0.0;
        }
        eigval_range[0] = eigval_min;
        eigval_range[1] = eigval_max;
        n_ev_range = NULL;
    }

    D_INT neigs = 0; // Note this will be set by Heev

    error = Heev(matA, 'U', n_ev_range, eigval_range, eig_vals, Deig_vecs, &neigs);
    if (error)
    {
        goto end_BSE_Solver1;
    }

    *neigs_found = neigs;

    if (Deig_vecs)
    {
        // FIX ME : check eige_vecs

        // back transform eigen vectors
        // [X1, X2] = [[I_n , 0] [0, -I_n]] @ Q @ L @ Z@\lambda^**-1/2
        error = BtEig_QLZ(matA, Ham_r, Deig_vecs, NULL, NULL);
        if (error)
        {
            goto end_BSE_Solver1;
        }

        D_INT descz[9];
        error = set_descriptor(Deig_vecs, descz);
        if (error)
        {
            goto end_BSE_Solver1;
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
end_BSE_Solver1:;
    free(Ham_r);
    // Bcast eigen values to all cpus
end_BSE_Solver0:;
    return error;
}

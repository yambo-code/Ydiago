// This file contains Generalized eigen solver
#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "../common/dtypes.h"
#include "../common/error.h"
#include "../common/min_max.h"
#include "../diago.h"
#include "../matrix/matrix.h"
#include "../solvers.h"
#include "scalapack_header.h"

Err_INT Geev(void* DmatA, D_Cmplx* eig_vals, void* Deig_vecsL, void* Deig_vecsR)
{
    // diagozalized any matrix
    /*  Unlike for hermitian case, there is no direct solver for non-hermitian.
        This function is constructed from the individual pieces that were
    implemented in scalapack library.

        ! Once can improve the accuracy by balancing with gebal, but
    unfortunaly, ! gebal is present only for real case (MKL has complex case
    though)

        The following three steps are performed in this function
        to diagonalize any non-hermitian matrix:
        1) Hessenberg reduction
        2) Schur decomposition
        3) Solve triangular eigen value problem
        Notes :
        Due to restriction from P?lahqr function, block size must be >= 6

    On output, when both Deig_vecsL and Deig_vecsR are requested,
    Note that the left and right eigenvectors are constructed such that
    the overlap matrix is identity
    */

    // do basic checks
    if (!eig_vals)
    {
        // Fatal error, return immediately.
        return ERR_NULL_PTR_BUFFER;
    }

    struct D_Matrix* matA = DmatA;
    struct D_Matrix* eig_vecsL = Deig_vecsL;
    struct D_Matrix* eig_vecsR = Deig_vecsR;

    Err_INT error = check_mat_diago(matA, false);
    D_INT err_code = 0;

    if (error)
    {
        return error;  // Fatal error, return immediately.
    }

    if (eig_vecsL)
    {
        error = check_mat_diago(eig_vecsL, false);
        if (error)
        {
            return error;  // Fatal error, return immediately.
        }

        // zero out the buffers
        error = set_zero(eig_vecsL);
        if (error)
        {
            return error;  // Fatal error, return immediately.
        }
    }

    if (eig_vecsR)
    {
        error = check_mat_diago(eig_vecsR, false);
        if (error)
        {
            return error;  // Fatal error, return immediately.
        }

        // zero out the buffers
        error = set_zero(eig_vecsR);
        if (error)
        {
            return error;  // Fatal error, return immediately.
        }
    }
    // FIX ME : We need to check if two matrices have same context, dims etc

    if (matA->block_size[0] < 6)
    {
        // Due to restriction from P?lahqr, block size must be >= 6
        error = INCOMPATIBLE_BLOCK_SIZE_ERR;
        return error;  // Fatal error, return immediately.
    }

    // zero out the eigen value buffer
    for (D_LL_INT i = 0; i < matA->gdims[0]; ++i)
    {
        eig_vals[i] = 0;
    }

    if (!matA->cpu_engage)
    {
        goto end_Geev;  // cpu not participating in diago
    }

    D_INT desca[9], descQ[9];

    error = set_descriptor(matA, desca);
    if (error)
    {
        goto end_Geev;
    }

    D_INT izero = 1;  // scalapack indices start from 1
    D_Cmplx* tau = calloc(matA->gdims[0], sizeof(*tau));

    if (!tau)
    {
        error = BUF_ALLOC_FAILED;
        goto end_Geev;
    }

    D_Cmplx work_tmp[4];
    D_INT lwork = -1;

    // 1) Hessenberg reduction

    // Query
    SL_FunCmplx(gehrd)(matA->gdims, &izero, matA->gdims, matA->data, &izero,
                       &izero, desca, tau, work_tmp, &lwork, &err_code);

    if (err_code)
    {
        error = SL_WORK_SPACE_ERROR;
        goto end_Geev1;
    }

    lwork = rint(creal(work_tmp[0]) * SL_WORK_QUERY_FAC);

    D_Cmplx* hess_work = calloc(lwork, sizeof(*hess_work));
    if (hess_work)
    {
        // Compute the Hessenberg reduction
        SL_FunCmplx(gehrd)(matA->gdims, &izero, matA->gdims, matA->data, &izero,
                           &izero, desca, tau, hess_work, &lwork, &err_code);
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }
    free(hess_work);
    if (error || err_code)
    {
        if (err_code)
        {
            error = SL_HESSENBERG_ERROR;
        }
        goto end_Geev1;
    }

    D_Cmplx cmplx_dummy = 0;
    D_Cmplx* Qmat = &cmplx_dummy;  // buffer to store similarity matrix

    if (Deig_vecsL || Deig_vecsR)
    {
        // eigenvectors are required
        // NM : In case both are requested only right eigenvectors
        // are computed and left are obtained by computing the inverse
        // of the right eigenvectors
        if (Deig_vecsR)
        {
            Qmat = eig_vecsR->data;
            // set the Qmat to I_n
            error = set_identity(eig_vecsR);
            if (error)
            {
                goto end_Geev1;
            }

            error = set_descriptor(eig_vecsR, descQ);
            if (error)
            {
                goto end_Geev1;
            }
        }
        else
        {
            Qmat = eig_vecsL->data;
            // set the Qmat to I_n
            error = set_identity(eig_vecsL);
            if (error)
            {
                goto end_Geev1;
            }

            error = set_descriptor(eig_vecsL, descQ);
            if (error)
            {
                goto end_Geev1;
            }
        }
        // construct the similarity matrix of the Hessenberg reduction
        // Query
        lwork = -1;
        SL_FunCmplx(unmhr)("R", "N", matA->gdims, matA->gdims, &izero,
                           matA->gdims, matA->data, &izero, &izero, desca, tau,
                           Qmat, &izero, &izero, descQ, work_tmp, &lwork,
                           &err_code);
        if (err_code)
        {
            error = SL_WORK_SPACE_ERROR;
            goto end_Geev1;
        }

        lwork = rint(creal(work_tmp[0]) * SL_WORK_QUERY_FAC);

        D_Cmplx* similarly_work = calloc(lwork, sizeof(*similarly_work));

        if (similarly_work)
        {
            SL_FunCmplx(unmhr)("R", "N", matA->gdims, matA->gdims, &izero,
                               matA->gdims, matA->data, &izero, &izero, desca,
                               tau, Qmat, &izero, &izero, descQ, similarly_work,
                               &lwork, &err_code);
        }
        else
        {
            error = BUF_ALLOC_FAILED;
        }
        free(similarly_work);
        if (error || err_code)
        {
            if (err_code)
            {
                error = SL_DIAGO_ERROR;
            }
            goto end_Geev1;
        }
    }
    else
    {
        // dummy
        memcpy(descQ, desca, sizeof(desca[0]) * 9);
    }

    // 2)  Schur decompositon
    D_INT wantt = 1;
    D_INT wantz = 1;

    if (!Deig_vecsL && !Deig_vecsR)
    {
        // eigen values only
        wantt = 0;
        wantz = 0;
    }

    // Force it to be strictly in upper hessenberg form i.e aij = 0 for j < i-1
    D_Cmplx cmplx_zero = 0.0;

    for (D_INT i = 1; i <= matA->gdims[0]; ++i)
    {
        for (D_INT j = 1; j <= matA->gdims[0]; ++j)
        {
            if (j < i - 1)
            {
                SL_FunCmplx(elset)(matA->data, &i, &j, desca, &cmplx_zero);
            }
        }
    }

    D_INT iwork_tmp[4];

    D_INT ilwork = -1;
    lwork = -1;
    // Query
    SL_FunCmplx(lahqr)(&wantt, &wantz, matA->gdims, &izero, matA->gdims,
                       matA->data, desca, eig_vals, matA->gdims, &izero, Qmat,
                       descQ, work_tmp, &lwork, iwork_tmp, &ilwork, &err_code);

    if (err_code)
    {
        error = SL_WORK_SPACE_ERROR;
        goto end_Geev1;
    }

    lwork = rint(creal(work_tmp[0]) * SL_WORK_QUERY_FAC);
    ilwork = iwork_tmp[0];

    D_INT* iwork = malloc(sizeof(*iwork) * ilwork);
    D_Cmplx* schur_work = malloc(sizeof(*schur_work) * lwork);

    if (iwork && schur_work)
    {
        // Compute the schur decompositon to the eigen values
        SL_FunCmplx(lahqr)(&wantt, &wantz, matA->gdims, &izero, matA->gdims,
                           matA->data, desca, eig_vals, &izero, matA->gdims,
                           Qmat, descQ, schur_work, &lwork, iwork, &ilwork,
                           &err_code);
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }

    free(schur_work);
    free(iwork);

    if (error || err_code)
    {
        if (err_code)
        {
            error = SL_SCHUR_ERROR;
        }
        goto end_Geev1;
    }

    if (!Deig_vecsL && !Deig_vecsR)
    {
        // if donot want eigen vectors, skip the rest
        goto end_Geev1;
    }

    // 3) compute the eigen vectors
    char side = 'R';
    if (!Deig_vecsR)
    {
        side = 'L';
    }

    char howmny = 'B';
    D_INT* select = &izero;

    D_Cmplx* vl = &cmplx_dummy;
    D_Cmplx* vr = &cmplx_dummy;

    D_INT descvl[9], descvr[9];

    D_INT mm_trevc = 0;
    if (Deig_vecsL)
    {
        mm_trevc = eig_vecsL->gdims[1];
        vl = eig_vecsL->data;
        error = set_descriptor(eig_vecsL, descvl);
        if (error)
        {
            goto end_Geev1;
        }
    }
    if (Deig_vecsR)
    {
        mm_trevc = eig_vecsR->gdims[1];
        vr = eig_vecsR->data;
        error = set_descriptor(eig_vecsR, descvr);
        if (error)
        {
            goto end_Geev1;
        }
    }

    D_Cmplx* trevc_work = malloc(sizeof(*trevc_work) * 4 * desca[8]);
    D_float* trevc_rwork = malloc(sizeof(*trevc_rwork) * 4 * desca[8]);

    D_INT neigs = matA->gdims[0];

    if (trevc_work && trevc_rwork)
    {
        SL_FunCmplx(trevc)(&side, &howmny, select, matA->gdims, matA->data,
                           desca, vl, descvl, vr, descvr, &mm_trevc,
                           matA->gdims, trevc_work, trevc_rwork, &err_code);

        if (!err_code && Deig_vecsR)
        {
            // Normalize the right eigen vectors with euclidian norm
            for (D_INT i = 0; i < neigs; ++i)
            {
                D_float alpha = 0.0;
                D_INT jx = i + 1;
                // Compute the norm
                SLvec_norm2(eig_vecsR->gdims, &alpha, eig_vecsR->data, &izero,
                            &jx, descvr, &izero);
                // Sanity check
                if (fabs(alpha) < 1e-8)
                {
                    // This is an error
                    continue;
                }
                D_Cmplx beta = 1.0 / alpha;
                // scale
                SL_FunCmplx(scal)(eig_vecsR->gdims, &beta, eig_vecsR->data,
                                  &izero, &jx, descvr, &izero);
            }
            // compute the left eigenvectors from right Left = (Right)^-H
            // NM : We compute the left evs from inverse of right evs instead of
            // computing from p?trevc to avoid non-identity overlap.
            if (Deig_vecsL)
            {
                // Note, unlike Right eigenvectors, left eigenvectors are not
                // normalized
                D_Cmplx beta = 0.0;
                D_Cmplx alpha_one = 1.0;
                // store L = R^H
                SL_FunCmplx(geadd)("C", eig_vecsR->gdims, eig_vecsR->gdims + 1,
                                   &alpha_one, vr, &izero, &izero, descvr,
                                   &beta, vl, &izero, &izero, descvl);
                // Compute inverse of R^H
                error = Inverse_Dmat(Deig_vecsL);
            }
        }
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }
    free(trevc_rwork);
    free(trevc_work);
    if (err_code)
    {
        error = SL_TREVC_ERROR;
    }

end_Geev1:;
    free(tau);
end_Geev:;
    // Bcast all the eigen values
    err_code =
        MPI_Bcast(eig_vals, matA->gdims[0], D_Cmplx_MPI_TYPE, 0, matA->comm);
    if (!error && err_code)
    {
        // return the true error
        error = DIAGO_MPI_ERROR;
    }
    return error;
}

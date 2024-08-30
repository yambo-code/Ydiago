// This file contains Generalized eigen solver

#include "../solvers.h"
#include "../matrix/matrix.h"
#include "../SL/scalapack_header.h"

Err_INT Inverse_Dmat(void* Dmat)
{
    /*
    Inverts a distributed matrix
    */

    struct D_Matrix* matA = Dmat;

    Err_INT error = check_mat_diago(matA, false);
    D_INT err_code = 0;
    D_INT izero = 1; // scalapack indices start from 1

    if (error)
    {
        return error; // Fatal error, return immediately.
    }

    D_INT desca[9];
    error = set_descriptor(matA, desca);
    if (error)
    {
        goto end_inv;
    }

    if (!matA->cpu_engage)
    {
        goto end_inv; // cpu not participating in diago
    }

    // 1) compute LU decomposition
    D_INT* ipiv = calloc(matA->ldims[0] + matA->block_size[0] + 1, sizeof *ipiv);
    if (ipiv)
    {
        SL_FunCmplx(getrf)(matA->gdims, matA->gdims + 1, matA->data,
                           &izero, &izero, desca, ipiv, &err_code);
        if (err_code)
        {
            error = SL_LU_ERROR;
        }
        else
        {
            // 2) find the inverse

            // make a query
            D_INT lwork = -1, liwork = -1;
            D_Cmplx work_tmp[3];
            D_INT iwork_tmp[3];
            //
            SL_FunCmplx(getri)(matA->gdims, matA->data, &izero, &izero,
                               desca, ipiv, work_tmp, &lwork, iwork_tmp, &liwork, &err_code);
            //
            lwork = rint(creal(work_tmp[0]) * SL_WORK_QUERY_FAC);
            liwork = iwork_tmp[0];
            //
            D_Cmplx* work = malloc(sizeof(*work) * lwork);
            D_INT* iwork = malloc(sizeof(*iwork) * liwork);
            //
            if (work && iwork)
            {
                // perform the inverse
                SL_FunCmplx(getri)(matA->gdims, matA->data, &izero, &izero,
                                   desca, ipiv, work, &lwork, iwork, &liwork, &err_code);
                if (err_code)
                {
                    error = SL_TRI_INV_ERROR;
                }
            }
            else
            {
                error = BUF_ALLOC_FAILED;
            }
            free(work);
            free(iwork);
        }
    }
    else
    {
        error = BUF_ALLOC_FAILED;
    }
    free(ipiv);

end_inv:;
    return error;
}

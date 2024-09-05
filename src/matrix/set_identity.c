#include "matrix.h"
#include "../SL/scalapack_header.h"
#include "../diago.h"
#include "../common/error.h"
#include "../common/dtypes.h"

Err_INT set_identity(void* DmatA)
{
    // set DmatA to identity matrix

    Err_INT error = check_mat_diago(DmatA, false);
    if (error)
    {
        return error;
    }

    struct D_Matrix* matA = DmatA;

    if (!matA->cpu_engage)
    {
        return DIAGO_SUCCESS;
    }

    D_LL_INT nloc_elements = matA->ldims[0] * matA->ldims[1];

    // first set all elements to 0
    for (D_LL_INT i = 0; i < nloc_elements; ++i)
    {
        matA->data[i] = 0.0;
    }

    // set the diagonal to 1
    for (D_LL_INT i = 0; i < matA->gdims[0]; ++i)
    {
        // get the processor id of (i,i) element
        D_INT prow = INDXG2P(i, matA->block_size[0], 0, 0, matA->pgrid[0]);
        D_INT pcol = INDXG2P(i, matA->block_size[1], 0, 0, matA->pgrid[1]);

        if (prow == matA->pids[0] && pcol == matA->pids[1])
        {
            // compute the local indices
            D_INT iloc = INDXG2L(i, matA->block_size[0], prow, 0, matA->pgrid[0]);
            D_INT jloc = INDXG2L(i, matA->block_size[1], pcol, 0, matA->pgrid[1]);

            matA->data[iloc * matA->lda[0] + jloc * matA->lda[1]] = 1.0;
        }
    }

    return DIAGO_SUCCESS;
}

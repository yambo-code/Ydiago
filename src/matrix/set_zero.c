#include "matrix.h"

Err_INT set_zero(void* DmatA)
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

    return DIAGO_SUCCESS;
}
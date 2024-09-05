// Checks matrices are good to go
#include "matrix.h"
#include "../diago.h"
#include "../common/error.h"
#include "../common/dtypes.h"

Err_INT check_mat_diago(void* D_mat, bool even_check)
{
    // checks if the matrix provided is valid for diagonalization
    if (!D_mat)
    {
        return MATRIX_NOT_INIT;
    }
    struct D_Matrix* matA = D_mat;

    if (!matA->cpu_engage)
    {
        return DIAGO_SUCCESS; // these cpus donot participate
    }

    if (matA->gdims[0] != matA->gdims[1])
    {
        return MATRIX_NOT_SQUARE;
    }

    if (matA->block_size[0] != matA->block_size[1])
    {
        return EQUAL_BLOCK_SIZE_ERROR;
    }
    if (even_check)
    {
        if (matA->gdims[0] % 2)
        {
            return WRONG_NON_TDA_BSE_MAT;
        }
    }

    return DIAGO_SUCCESS;
}

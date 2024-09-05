#include "matrix.h"
#include "../diago.h"
#include "../common/error.h"
#include "../common/dtypes.h"

void descinit_(D_INT*, D_INT*, D_INT*, D_INT*, D_INT*, D_INT*, D_INT*, D_INT*, D_INT*, D_INT*);

Err_INT set_descriptor(void* D_mat, D_INT* desc)
{
    // sets the desc array. desc should be 9 length array

    if (!D_mat)
    {
        return MATRIX_NOT_INIT; // error.
    }
    struct D_Matrix* mat = D_mat;

    if (!mat->cpu_engage)
    {
        // this cpu is not participating
        return DIAGO_SUCCESS;
    }
    // Incase cpu is participating, but desc is NULL, then return with an error
    if (!desc)
    {
        return DESC_ERROR;
    }

    D_INT err_info;
    D_INT izero = 0;
    D_INT lda = mat->lda[0] > mat->lda[1] ? mat->lda[0] : mat->lda[1];

    descinit_(desc, mat->gdims, mat->gdims + 1, mat->block_size,
              mat->block_size + 1, &izero, &izero, &mat->blacs_ctxt,
              &lda, &err_info);

    if (err_info)
    {
        return DESC_ERROR;
    }
    return DIAGO_SUCCESS;
}

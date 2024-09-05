#include "matrix.h"
#include "../SL/scalapack_header.h"
#include "../diago.h"
#include <mpi.h>
#include "../common/error.h"
#include "../common/dtypes.h"
#include <stdlib.h>
#include <math.h>
#include <complex.h>

Err_INT Construct_BSE_RealHam(void* DmatA, D_float* matA_out)
{
    // All cpus must call this function.
    /*
    Given a (2n,2n) matrix of form
    |  A    B   |
    | -B*  -A*  |
    in block cyclic layout,
    This function constructs a real hamilition of dimension (2n,2n), which
    is equal to
    |  Re(A+B)    Im(A-B) |
    | -Im(A+B)    Re(A-B) |
    and stores in matA_out in the same block cyclic layout

    See : https://doi.org/10.1016/j.laa.2015.09.036
    for more details.

    Note matA_out and DmatA must have identical layout
    */

    Err_INT error = check_mat_diago(DmatA, true);
    if (error)
    {
        return error;
    }

    struct D_Matrix* matA = DmatA;

    if (!matA->cpu_engage)
    {
        return DIAGO_SUCCESS; // these cpus donot participate
    }

    D_LL_INT nloc_elem = matA->ldims[0] * matA->ldims[1];

    D_float* buf1 = calloc(nloc_elem + 1, sizeof(*buf1));
    if (!buf1)
    {
        error = BUF_ALLOC_FAILED;
        goto err_bse_RHam_0;
    }

    D_float* buf2 = calloc(nloc_elem + 1, sizeof(*buf2));
    if (!buf2)
    {
        error = BUF_ALLOC_FAILED;
        goto err_bse_RHam_1;
    }
    // add +1 buffer to avoid implementation defined behaviour

    /*
    buf1 should get
    |  Re(A)    Im(A) |
    | -Im(A)    Re(A) |

    buf2 should get
    | -Re(B)    Im(B) |
    |  Im(B)    Re(B) |

    result = buf1-buf2
    */

    // first lets fill real
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        matA_out[i] = creal(matA->data[i]);
    }

    D_INT ndim = matA->gdims[0] / 2;

    D_INT ia_from = 1;
    D_INT ja_from = 1;

    D_INT ib_to = 1;
    D_INT jb_to = 1;

    D_INT desca[9];

    error = set_descriptor(matA, desca);
    if (error)
    {
        goto err_bse_RHam_2;
    }

    D_INT ictxt = matA->blacs_ctxt;

    // set real(A) in buf1

    ia_from = 1;
    ja_from = 1;
    ib_to = 1;
    jb_to = 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf1, &ib_to,
                        &jb_to, desca, &ictxt);

    ia_from = 1;
    ja_from = 1;
    ib_to = ndim + 1;
    jb_to = ndim + 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf1, &ib_to,
                        &jb_to, desca, &ictxt);

    // set real(B) in buf2
    ia_from = 1;
    ja_from = ndim + 1;
    ib_to = ndim + 1;
    jb_to = ndim + 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf2, &ib_to,
                        &jb_to, desca, &ictxt);

    // negate and set the -Re(B) in buf2
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        matA_out[i] = -matA_out[i];
    }

    ia_from = 1;
    ja_from = ndim + 1;
    ib_to = 1;
    jb_to = 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf2, &ib_to,
                        &jb_to, desca, &ictxt);

    // set imaginary parts
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        matA_out[i] = cimag(matA->data[i]);
    }

    // fill imag(B) in buf2
    ia_from = 1;
    ja_from = ndim + 1;
    ib_to = 1;
    jb_to = ndim + 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf2, &ib_to,
                        &jb_to, desca, &ictxt);

    ia_from = 1;
    ja_from = ndim + 1;
    ib_to = ndim + 1;
    jb_to = 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf2, &ib_to,
                        &jb_to, desca, &ictxt);

    // fill rest of buf1
    ia_from = 1;
    ja_from = 1;
    ib_to = 1;
    jb_to = ndim + 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf1, &ib_to,
                        &jb_to, desca, &ictxt);

    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        matA_out[i] = -matA_out[i];
    }

    ia_from = 1;
    ja_from = 1;
    ib_to = ndim + 1;
    jb_to = 1;
    SL_FunFloat(gemr2d)(&ndim, &ndim, matA_out,
                        &ia_from, &ja_from, desca, buf1, &ib_to,
                        &jb_to, desca, &ictxt);

    // set mat out
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        // matout = buf1-buf2
        matA_out[i] = buf1[i] - buf2[i];
    }

err_bse_RHam_2:
    free(buf2);
err_bse_RHam_1:
    free(buf1);
err_bse_RHam_0:
    return error;
}

#include "matrix.h"

Err_INT BtEig_QLZ(void* DmatA, D_float* Lmat, void* DmatZ, D_INT neig, char* gpu, void* einfo)
{
    /*
    For now: this function (GEMM operation) is  not gpu ported
    do a back transformation of bse eigen vectors i.e
    Z = [[I_n , 0] [0, -I_n]] @ Q @ L @ Z
    where Q = [[I_n  -i*I_n], [I_n  i*I_n]]

    Z on input contains eigen vectors
    on output, Z gets replaced with back transformed eigen vectors (not normalized)
    DmatA and Lmat will be destroyed

    GPU and einfo (elpainfo) are not used.
    
    DmatZ contains neig eigenvectors arranged in coloumns starting from 1
    */

    Err_INT error = check_mat_diago(DmatA, true);
    if (error)
    {
        goto err_bt_0;
    }

    if (!DmatZ)
    {
        error = MATRIX_INIT_FAILED;
        goto err_bt_0;
    }

    struct D_Matrix* matA = DmatA;
    struct D_Matrix* matZ = DmatZ;

    if (matA->cpu_engage && !Lmat)
    { // error
        error = ERR_NULL_PTR_BUFFER;
        goto err_bt_0;
    }

    // ! FIX ME : MN We need to check if parameters of two dmats are same

    D_LL_INT nloc_elem = matA->ldims[0] * matA->ldims[1];

    D_INT ndim = matA->gdims[0] / 2;

    D_INT desca[9], descz[9];

    error = set_descriptor(matA, desca);
    if (error)
    {
        goto err_bt_0;
    }

    error = set_descriptor(matZ, descz);
    if (error)
    {
        goto err_bt_0;
    }

    D_INT ictxt = matA->blacs_ctxt;

    D_float* buf_real = calloc(nloc_elem + 1, sizeof(*buf_real));
    if (!buf_real)
    {
        error = BUF_ALLOC_FAILED;
        goto err_bt_0;
    }
    D_float* buf_imag = calloc(nloc_elem + 1, sizeof(*buf_imag));
    if (!buf_imag)
    {
        error = BUF_ALLOC_FAILED;
        goto err_bt_1;
    }
    // set the buffer to 0
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        buf_real[i] = 0.0;
        buf_imag[i] = 0.0;
    }

    /*
    1) compute [[I_n , 0] [0, -I_n]] @ Q @ L
    */
    if (matA->cpu_engage)
    {
        D_INT ia_from = 1;
        D_INT ja_from = 1;

        D_INT ib_to = 1;
        D_INT jb_to = 1;

        // set real part

        // first L1 block
        SL_FunFloat(trmr2d)("L", "N", &ndim, &ndim,
                            Lmat, &ia_from, &ja_from, desca,
                            buf_real, &ib_to, &jb_to, desca, &ictxt);

        // negate
        for (D_LL_INT i = 0; i < nloc_elem; ++i)
        {
            Lmat[i] = -Lmat[i];
        }

        ia_from = 1;
        ja_from = 1;
        ib_to = ndim + 1;
        jb_to = 1;

        SL_FunFloat(trmr2d)("L", "N", &ndim, &ndim,
                            Lmat, &ia_from, &ja_from, desca,
                            buf_real, &ib_to, &jb_to, desca, &ictxt);

        // copy imag part

        // L2 blocks
        ia_from = ndim + 1;
        ja_from = ndim + 1;
        ib_to = 1;
        jb_to = ndim + 1;

        SL_FunFloat(trmr2d)("L", "N", &ndim, &ndim,
                            Lmat, &ia_from, &ja_from, desca,
                            buf_imag, &ib_to, &jb_to, desca, &ictxt);

        ia_from = ndim + 1;
        ja_from = ndim + 1;
        ib_to = ndim + 1;
        jb_to = ndim + 1;

        SL_FunFloat(trmr2d)("L", "N", &ndim, &ndim,
                            Lmat, &ia_from, &ja_from, desca,
                            buf_imag, &ib_to, &jb_to, desca, &ictxt);

        // finally copy the L12 block
        ia_from = ndim + 1;
        ja_from = 1;
        ib_to = 1;
        jb_to = 1;

        SL_FunFloat(gemr2d)(&ndim, &ndim, Lmat,
                            &ia_from, &ja_from, desca, buf_imag, &ib_to,
                            &jb_to, desca, &ictxt);

        ia_from = ndim + 1;
        ja_from = 1;
        ib_to = ndim + 1;
        jb_to = 1;

        SL_FunFloat(gemr2d)(&ndim, &ndim, Lmat,
                            &ia_from, &ja_from, desca, buf_imag, &ib_to,
                            &jb_to, desca, &ictxt);
    }
    D_float factor = 1.0;
    D_INT izero = 1;

    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        matA->data[i] = buf_real[i] + factor * I * buf_imag[i];
    }
    free(buf_real);
    free(buf_imag);

    D_LL_INT nloc_elem_Z = matZ->ldims[0] * matZ->ldims[1];

    D_Cmplx* z_tmp = malloc((nloc_elem_Z + 1) * sizeof(*z_tmp));
    if (!z_tmp)
    {
        error = BUF_ALLOC_FAILED;
        goto err_bt_0;
    }

    memcpy(z_tmp, matZ->data, nloc_elem_Z * sizeof(*z_tmp));

    D_Cmplx alpha_one = 1.0;
    D_Cmplx beta_zero = 0.0;

    if (matA->cpu_engage)
    {
        // This should be ported to gpus. (elpa has it), but not sure
        // if it will give any millage
        SL_FunCmplx(gemm)("N", "N", matA->gdims, &neig,
                          matA->gdims + 1, &alpha_one, matA->data, &izero,
                          &izero, desca, z_tmp, &izero,
                          &izero, descz, &beta_zero, matZ->data,
                          &izero, &izero, descz);
    }

    free(z_tmp);

    return error;

// error gotos
err_bt_1:
    free(buf_real);
err_bt_0:
    return error;
}

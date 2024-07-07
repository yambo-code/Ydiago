#include "matrix.h"
// builds the skew symmetric matrix required when diagonalizing non-TDA bse
// hamiliton.

Err_INT Symplectic_times_L(void* DmatA, D_float* Lmat, D_float* out_Omega_L)
{
    /*
    If L is lower triangular matrix, then this function returns
    \Omega @ L where \Omega is Symplectic matrix |0    I_n|
    -------------------------------------------  |-I_n   0|
    DmatA : is the original BSE hamiliton (only used to read BLACS variables)
    L : on input contains L
    out_Omega_L contains the output \Omega @ L
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

    // set the buffer to 0
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        out_Omega_L[i] = 0.0;
    }

    /*
    -------| L1    0|
    L =    | L12  L2|
    -------|--------|
    */

    D_INT ndim = matA->gdims[0] / 2;

    D_INT desca[9];
    error = set_descriptor(matA, desca);
    if (error)
    {
        return error;
    }

    D_INT ictxt = matA->blacs_ctxt;

    D_INT ia_from = ndim + 1;
    D_INT ja_from = 1;

    D_INT ib_to = 1;
    D_INT jb_to = 1;

    // first copy the L12 block
    SL_FunFloat(gemr2d)(&ndim, &ndim, Lmat,
                        &ia_from, &ja_from, desca, out_Omega_L, &ib_to,
                        &jb_to, desca, &ictxt);

    // copy the L2
    ia_from = ndim + 1;
    ja_from = ndim + 1;
    ib_to = 1;
    jb_to = ndim + 1;

    SL_FunFloat(trmr2d)("L", "N", &ndim, &ndim,
                        Lmat, &ia_from, &ja_from, desca,
                        out_Omega_L, &ib_to, &jb_to, desca, &ictxt);

    // copy -L1
    // first negate the matrix
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
                        out_Omega_L, &ib_to, &jb_to, desca, &ictxt);

    // undo the negative sign
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        Lmat[i] = -Lmat[i];
    }

    return DIAGO_SUCCESS;
}

Err_INT Construct_bseW(void* DmatA, D_float* Lmat, D_float* Wmat, char* gpu, void* einfo)
{
    /* This routine constructs W = L^T * |0    I_n| * L where L is the lower
    ------------------------------------ |-I_n   0| ------------------------
    triangular matrix obtained from Cholesky factorization of
    |  Re(A+B)    Im(A-B) |
    | -Im(A+B)    Re(A-B) |
    */

    /*
    DmatA : is the original BSE hamiliton (only used to read BLACS variables)
    Lmat :  contains L  (input)
    Wmat : output contains W (output)
    */
    // compute \omega \time L, where \omega is Symplectic matrix
    // einfo is elpa hander. only refernced when compiled with GPU flag

    /*
    GPU and einfo (elpainfo) are not used.
    *** This function (GEMM operation) is not GPU ported.

    */

    Err_INT error = check_mat_diago(DmatA, true);

    if (error)
    {
        return error;
    }

    struct D_Matrix* matA = DmatA;

    if (matA->cpu_engage)
    {
        if (!DmatA || !Lmat || !Wmat)
        { // error
            return MATRIX_NOT_INIT;
        }
    }

    error = Symplectic_times_L(DmatA, Lmat, Wmat);

    if (error)
    {
        return error;
    }

    D_LL_INT nloc_elem = matA->ldims[0] * matA->ldims[1];

    // perform the Gemm
    D_INT izero = 1;
    D_float alpha = 1.0;
    D_INT desca[9];
    error = set_descriptor(matA, desca);

    if (error)
    {
        return error;
    }

    if (matA->cpu_engage)
    {
        // This should be ported to gpus. (elpa has it), but not sure
        // if it will give any millage.
        SL_FunFloat(trmm)("L", "L", "T",
                          "N", matA->gdims, matA->gdims,
                          &alpha, Lmat, &izero,
                          &izero, desca, Wmat,
                          &izero, &izero, desca);
    }

    return error;
}

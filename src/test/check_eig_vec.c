// function to check eigen vectors and see if diagonalizatin is correct
#include "tests.h"
#include <stdbool.h>

void print_loc(void* D_Mat)
{
    struct D_Matrix* mat = D_Mat;
    for (int i = 0; i < mat->ldims[0]; ++i)
    {
        for (int j = 0; j < mat->ldims[1]; ++j)
        {
            D_Cmplx tmp_d = mat->data[i * mat->lda[0] + j * mat->lda[1]];
            printf("(%.4f+%.4fj),  ", creal(tmp_d), cimag(tmp_d));
        }
        printf("\n");
    }
}

void copy_mats(void* des_mat, void* src_mat)
{
    struct D_Matrix* des = des_mat;
    struct D_Matrix* src = src_mat;

    if (des->pids[0] < 0 || des->pids[1] < 0)
    {
        return;
    }
    D_LL_INT nloc = src->ldims[0] * src->ldims[1];

    memcpy(des->data, src->data, sizeof(*src->data) * nloc);

    return;
}

D_float check_eig_vecs(void* D_mat, D_Cmplx* eig_vals,
                       void* Deig_vecs, const D_INT neigs)
{
    struct D_Matrix* matA = D_mat;
    struct D_Matrix* eig_vecs = Deig_vecs;

    D_float sum = 0.0;
    bool passed = true;
    if (matA->pids[0] < 0 || matA->pids[1] < 0)
    {
        goto chech_eig_end;
    }

    D_LL_INT nloc = matA->ldims[0] * matA->ldims[1];

    D_Cmplx* bufAZ = calloc(nloc, sizeof(*bufAZ));
    check_ptr(bufAZ);

    D_Cmplx* bufZAZ = calloc(nloc, sizeof(*bufZAZ));
    check_ptr(bufZAZ);

    D_INT desca[9], descz[9];

    int error = set_descriptor(matA, desca);
    check_error(error);

    error = set_descriptor(eig_vecs, descz);
    check_error(error);

    D_INT izero = 1;

    D_Cmplx alphas = 0.0;
    // set the eigevalues which are not computed to zero
    for (D_INT jx = neigs + 1; jx <= eig_vecs->gdims[0]; ++jx)
    {
        SL_FunCmplx(scal)(eig_vecs->gdims, &alphas, eig_vecs->data, &izero, &jx, descz, &izero);
    }
    // A*Z (n,n) (n,z)
    D_Cmplx alpha = 1.0;
    D_Cmplx beta = 0.0;

    SL_FunCmplx(gemm)("N", "N", matA->gdims, matA->gdims,
                      matA->gdims, &alpha, matA->data, &izero,
                      &izero, desca, eig_vecs->data, &izero,
                      &izero, descz, &beta, bufAZ,
                      &izero, &izero, descz);

    SL_FunCmplx(gemm)("C", "N", matA->gdims, matA->gdims,
                      matA->gdims, &alpha, eig_vecs->data, &izero,
                      &izero, descz, bufAZ, &izero,
                      &izero, descz, &beta, bufZAZ,
                      &izero, &izero, descz);

    // set the diagonal to 1
    for (D_LL_INT i = 0; i < neigs; ++i)
    {
        // get the processor id of (i,i) element
        D_INT prow = INDXG2P(i, matA->block_size[0], 0, 0, matA->pgrid[0]);
        D_INT pcol = INDXG2P(i, matA->block_size[1], 0, 0, matA->pgrid[1]);

        if (prow == matA->pids[0] && pcol == matA->pids[1])
        {
            // compute the local indices
            D_INT iloc = INDXG2L(i, matA->block_size[0], prow, 0, matA->pgrid[0]);
            D_INT jloc = INDXG2L(i, matA->block_size[1], pcol, 0, matA->pgrid[1]);

            bufZAZ[iloc * matA->lda[0] + jloc * matA->lda[1]] -= eig_vals[i];
        }
    }

    for (D_LL_INT i = 0; i < nloc; ++i)
    {
        D_float abbb = cabs(bufZAZ[i]);
        sum += abbb * abbb;
    }

    free(bufAZ);
    free(bufZAZ);
chech_eig_end:;

    D_float sum_net = 0;
    MPI_Allreduce(&sum, &sum_net, 1, D_float_MPI_TYPE, MPI_SUM, matA->comm);

    sum_net = sqrt(sum_net);

    return sum_net;
}

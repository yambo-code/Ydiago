// Contain Special non-TDA BSE Solver
#include "../solvers.h"
#include "../matrix/matrix.h"
#include "../SL/scalapack_header.h"
#include "elpa_wrap.h"

#ifdef WITH_ELPA

Err_INT BSE_Solver_Elpa(void* D_mat, D_Cmplx* eig_vals, void* Deig_vecs,
                        D_INT neigs, const D_INT elpa_solver,
                        char* gpu_type, const D_INT nthreads)
{
    /*
    The Matrix must have the the following structure
    |  A    B   |
    | -B*  -A*  |
    Where A is hermitian and B is symmetric
    And moreover.
    |  A    B   |
    |  B*   A*  | must be postive definite.
    */
    // This method is based on
    // https://doi.org/10.1016/j.laa.2015.09.036

    /*

    DmatA is destroyed

    Eigen values come in pair i.e (-lambda, lambda).
    Only computes postive eigenvalues and their correspoing right
    eigen vectors. From this we can retreive left eigen vectors
    for positive eigen values, and left and right eigenvectors of negative
    eigen values

        Right eigenvectors         Left eigenvectors
          +ve     -ve               +ve       -ve
    X = [ X_1, conj(X_2) ]    Y = [  X_1, -conj(X_2)]
*        [X_2, conj(X_1) ]        [ -X_2,  conj(X_1)],

    The code returns [X1,X2] as eigen vectors
    */

    int error = 0;
    int mpi_error = 0;

    if (!D_mat)
    { // error
        return -1;
    }

    struct D_Matrix* matA = D_mat;
    struct D_Matrix* evecs = Deig_vecs;

    if (matA->gdims[0] != matA->gdims[1])
    {
        return -1; // error not a square matrix
    }

    if (matA->block_size[0] != matA->block_size[1])
    {
        return -1; // error block size must be square
    }

    if (matA->gdims[0] % 2)
    {
        return -1; // the dimension must be even
    }

    D_INT ndim = matA->gdims[0] / 2;

    if (neigs < 1 || neigs > ndim)
    {
        // errornous number of eigen values. compute all of them
        neigs = ndim;
    }

    // we need to create a comm that participates in diagonalization
    MPI_Comm diago_comm;
    int colour_diago_comm = 0;

    if (matA->pids[0] < 0 || matA->pids[1] < 0)
    {
        colour_diago_comm = 1;
    }

    int my_rank_comm;
    MPI_Comm_rank(matA->comm, &my_rank_comm);

    mpi_error = MPI_Comm_split(matA->comm, colour_diago_comm, my_rank_comm, &diago_comm);

    if (colour_diago_comm)
    {
        goto end_BSE_Solver_elpa;
    }

    elpa_t elpa_handle;
    error = set_ELPA(D_mat, neigs, elpa_solver, gpu_type, nthreads, diago_comm, &elpa_handle);

    D_LL_INT nloc_elem = matA->ldims[0] * matA->ldims[1];

    D_float* Ham_r = calloc(nloc_elem + 1, sizeof(*Ham_r));

    D_INT izero = 1;

    D_INT desca[9];
    if (set_descriptor(matA, desca))
    {
        return -3;
    }

    // 1) compute the real hamilitian
    error = Construct_BSE_RealHam(matA, Ham_r);

    // 2) Perform the Cholesky factorization for real symmetric matrix
    ElpaFloat(cholesky)(handle, Ham_r, &error)
        // L is stotred in Ham
        if (error)
    {
        // Cholesky factorization failed, not a positive definte matrix
        return -100;
    }

    // 3) W = L^T \omega * L
    /*
    Note that this is real. and W is skew symmetric.
    */
    D_float* Wmat = calloc(nloc_elem + 1, sizeof(*Wmat));

    error = Construct_bseW(matA, Ham_r, Wmat, gpu_type);

    /*
    MN : FIX ME : For elpa, there is a skew symmetric solver
    use it instead of hermitian solver
    */

    // compute -iW and diagonalize using hermitian solver (only positive eigen values)
    for (D_LL_INT i = 0; i < nloc_elem; ++i)
    {
        // Note we diagonalize iW instead of -iW because, there is no way
        // to get the upper part of the spectra without full diagnoalization.
        // Due to this, all eigen values are computed with ELPA.
        // We compute the negative eigenvalues of iW (which are +ve eigs of -iW)
        matA->data[i] = I * Wmat[i];
    }

    free(Wmat);

    // set all elements to Zero in eig_vecs
    for (D_LL_INT i = 0; i < evecs->ldims[0] * evecs->ldims[1]; ++i)
    {
        evecs->data[i] = 0;
    }

    error = Heev_Elpa(D_mat, eig_vals, Deig_vecs, neigs, elpa_solver, gpu_type, nthreads);

    // back transform eigen vectors
    // [X1, X2] = [[I_n , 0] [0, -I_n]] @ Q @ L @ Z@\lambda^**-1/2
    error = BtEig_QLZ(matA, Ham_r, Deig_vecs, gpu_type);

    D_INT descz[9];
    if (set_descriptor(Deig_vecs, descz))
    {
        return -3;
    }

    // normalize the eigen vectors with |2*lambda|**-0.5
    for (D_INT i = 0; i < neigs; ++i)
    {
        // first negate the eigenvalues, because we computed for iW (instead of -iW)
        eig_vals[i] = -eig_vals[i];

        D_Cmplx alpha = csqrt(2 * cabs(eig_vals[i]));

        if (cabs(alpha) < 1e-8)
        {
            // FIX ME : NM: This is an error because we only want eigen values >0
            continue;
        }
        else
        {
            alpha = 1.0 / alpha;
        }
        D_INT jx = i + 1;

        SL_FunCmplx(scal)(evecs->gdims, &alpha, evecs->data, &izero, &jx, descz, &izero);
    }

    free(Ham_r);

    error = cleanup_ELPA(elpa_handle);

end_BSE_Solver_elpa:;

    // free the new comm
    mpi_error = MPI_Comm_free(&diago_comm);

    // Bcast eigen values to all cpus
    mpi_error = MPI_Bcast(eig_vals, neigs, D_Cmplx_MPI_TYPE, 0, matA->comm);

    return 0;
}

#endif
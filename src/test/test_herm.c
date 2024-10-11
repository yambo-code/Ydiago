#include "check_eig_vec.c"
#include "load_file.c"
#include "tests.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    setbuf(stdout, NULL);

    int size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double start, end;

    // D_INT Grows = 4, Gcols = 4;
    // D_INT ProcX = 1, ProcY = 1;
    // D_INT blockX = 2, blockY = 2;

    D_INT Grows = 100, Gcols = 100;
    D_INT ProcX = 2, ProcY = 3;
    D_INT blockX = 8, blockY = 8;

    void* mpicxt = BLACScxtInit('R', MPI_COMM_WORLD, ProcX, ProcY);
    check_ptr(mpicxt);

    void* Matrix_A = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt);
    check_ptr(Matrix_A);

    void* Matrix_Z = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt);
    check_ptr(Matrix_Z);

    Err_INT error;

    D_Cmplx* eig_vals_ref = calloc(Grows, sizeof(*eig_vals_ref));
    check_ptr(eig_vals_ref);

    D_Cmplx* eig_vals = calloc(Grows, sizeof(*eig_vals));
    check_ptr(eig_vals);

    load_mat_file("test_mats/Herm_100.mat", "test_mats/Herm_100_eigs.mat",
                  Matrix_A, eig_vals_ref, false);  // bool bse_mat);

    void* Matrix_A_copy = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt);
    check_ptr(Matrix_A_copy);

    copy_mats(Matrix_A_copy, Matrix_A);

    D_INT nfound = Grows;

#ifdef TEST_ELPA
    if (!my_rank) printf("ELPA Heev\n");
    error = Heev_Elpa(Matrix_A, eig_vals, Matrix_Z, nfound, 2, NULL, 1);
#else
    // Geev(Matrix, eig_vals, NULL, Matrix_Z);
    if (!my_rank) printf("Scalapack Heev\n");
    D_INT range[2] = {1, Grows};
    error = Heev(Matrix_A, 'L', range, NULL, eig_vals, Matrix_Z, &nfound);
#endif
    check_error(error);

    bool passed = true;
    if (!my_rank)
    {
        printf("Eigenvalues found : %d .\n", nfound);
        for (D_INT i = 0; i < nfound; ++i)
        {
            D_float err_eig = 100 * cabs(eig_vals[i] - eig_vals_ref[i]) /
                              cabs(eig_vals_ref[i]);
            // printf("percentage Error : %f, Ref : %f, Calc : %f\n", err_eig,
            // cabs(eig_vals_ref[i]), cabs(eig_vals[i])  );
            if (err_eig > 2e-2)
            {
                printf("percentage Error : %f, Ref : %f, Calc : %f\n", err_eig,
                       cabs(eig_vals_ref[i]), cabs(eig_vals[i]));
                passed = false;
                break;
            }
        }
        if (passed)
            printf("Hermtian Eigenvalues test : Passed.\n");
        else
            printf("Hermtian Eigenvalues test : Failed.\n");
    }

    D_float eig_res_err =
        check_eig_vecs(Matrix_A_copy, eig_vals, Matrix_Z, nfound);
    if (!my_rank) printf("Error : %f \n", eig_res_err);

    free(eig_vals);
    free(eig_vals_ref);
    free_D_Matrix(Matrix_Z);
    free_D_Matrix(Matrix_A);
    free_D_Matrix(Matrix_A_copy);
    BLACScxtFree(mpicxt);

    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

#include "../diago.h"
#include "../solvers.h"

#ifdef NOPRINT
#define printf(A, ...) ;
#endif

#define PRINT_EIGS 1

void print_loc(void* D_Mat)
{
    struct D_Matrix* mat = D_Mat;
    for (int i = 0; i < mat->ldims[0]; ++i)
    {
        for (int j = 0; j < mat->ldims[1]; ++j)
        {
            D_Cmplx tmp_d = mat->data[i * mat->lda[0] + j * mat->lda[1]];
            printf("(%.1f  %.1f),  ", creal(tmp_d), cimag(tmp_d));
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    setbuf(stdout, NULL);

    int size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double start, end;

    D_INT Grows = 16, Gcols = 16;
    D_INT ProcX = 2, ProcY = 2;
    D_INT blockX = 8, blockY = 8;

    void* mpicxt = BLACScxtInit('R', MPI_COMM_WORLD, ProcX, ProcY);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    void* Matrix = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (!my_rank) printf("Init D mat : %f\n", end - start);

    if (!Matrix)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Err_INT error;

    D_LL_INT Total_elements = Grows * Gcols;
    D_LL_INT ele_this_cpu = Total_elements / size;
    D_LL_INT ele_rem = Total_elements % size;

    D_LL_INT shift = ele_this_cpu * my_rank;

    if (my_rank < ele_rem)
    {
        shift += my_rank;
        ++ele_this_cpu;
    }
    else
    {
        shift += ele_rem;
    }

    // initiate Queue
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    error = initiateSetQueue(Matrix, ele_this_cpu);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (!my_rank) printf("Init set : %f\n", end - start);

    if (error)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // set elements
    for (D_LL_INT iele = 0; iele < ele_this_cpu; ++iele)
    {
        D_LL_INT gele = iele + shift;
        D_LL_INT ir = gele / Gcols;
        D_LL_INT jc = gele % Gcols;
        D_Cmplx tmp11 = csin(gele + I * ccos(gele));

        gele = ir + jc * Gcols;
        tmp11 = tmp11 + conj(csin(gele + I * ccos(gele)));

        error = DMatSet(Matrix, ir, jc, tmp11);
        if (error)
        {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (!my_rank) printf("Setting mat : %f\n", end - start);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Process the Queue
    error = ProcessSetQueue(Matrix);

    end = MPI_Wtime();
    if (!my_rank) printf("SetQueue Process : %f\n", end - start);

    if (error)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // check if the elements in the local block are correct

    bool loc_correct = true;

    struct D_Matrix* dismat = Matrix;

    for (int iloc = 0; iloc < dismat->ldims[0]; ++iloc)
    {
        for (int jloc = 0; jloc < dismat->ldims[1]; ++jloc)
        {
            int iglob = INDXL2G(iloc, dismat->block_size[0], dismat->pids[0], 0,
                                dismat->pgrid[0]);
            int jglob = INDXL2G(jloc, dismat->block_size[1], dismat->pids[1], 0,
                                dismat->pgrid[1]);

            D_LL_INT gele = iglob * Gcols + jglob;
            D_Cmplx tmp11 = csin(gele + I * ccos(gele));
            gele = jglob * Gcols + iglob;
            tmp11 = tmp11 + conj(csin(gele + I * ccos(gele)));

            if (cabs(tmp11 - dismat->data[iloc * dismat->lda[0] +
                                          jloc * dismat->lda[1]]) > 1e-6)
            {
                loc_correct = false;
            };
        }
    }

    bool elem_pass = true;
    MPI_Reduce(&loc_correct, &elem_pass, 1, MPI_C_BOOL, MPI_LOR, 0,
               MPI_COMM_WORLD);
    //////print Local block
    // if (my_rank == 0)
    // {
    //     printf("******* Local block ******\n");
    //     print_loc(Matrix);
    // }

    if (!my_rank)
    {
        if (elem_pass)
        {
            printf("Local elements set correctly. Passed\n");
        }
        else
        {
            printf("Local elements NOT set correctly. Failed\n");
        }
    }

    void* Matrix_Z = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt);

    if (!Matrix_Z)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    D_Cmplx* eig_vals = calloc(Grows, sizeof(*eig_vals));

    if (!eig_vals)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    if (!my_rank) printf("Diagonalization.......\n");

#ifdef TEST_ELPA
    Heev_Elpa(Matrix, eig_vals, Matrix_Z, -1, 2, NULL, 1);
#else
    Geev(Matrix, eig_vals, NULL, Matrix_Z);
    D_INT nfound = 0;
    // if (Heev(Matrix, 'U', NULL, NULL, eig_vals, Matrix_Z, &nfound))
    // {
    //     MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    // }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (!my_rank) printf("Diagonalization time : %f\n", end - start);

#ifdef NOPRINT
#undef printf
#endif

    if (!my_rank)
    {
        D_float refvals[16] = {
            -1.01363347e+01, -8.85451267e+00, -4.73670736e+00, -4.41315701e+00,
            -9.16672860e-01, -1.27597110e-01, -2.06397617e-02, -7.57303011e-04,
            7.83462424e-04,  1.85921577e-02,  2.40734728e-01,  6.46603001e-01,
            4.23501051e+00,  4.78593026e+00,  7.46607179e+00,  9.85701459e+00};

#ifdef PRINT_EIGS
        for (D_INT ii = 0; ii < Grows; ++ii)
        {
            printf("%.10f %.10f \n", creal(eig_vals[ii]), cimag(eig_vals[ii]));
        }
#else
        bool pass = true;
        for (D_INT ii = 0; ii < Grows; ++ii)
        {
            if (fabs(creal(eig_vals[ii]) - refvals[ii]) > 1e-5 ||
                fabs(cimag(eig_vals[ii]) - 0.0) > 1e-8)
                pass = false;
            if (!pass) break;
        }

        if (pass)
            printf("Passed ;)\n");
        else
            printf("Failed ;)\n");
#endif
    }

    free(eig_vals);
    free_D_Matrix(Matrix_Z);
    free_D_Matrix(Matrix);
    BLACScxtFree(mpicxt);

    MPI_Finalize();
    return 0;
}

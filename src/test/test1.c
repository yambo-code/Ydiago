#include "../diago.h"
#include "../solvers.h"
#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

#ifdef NOPRINT
#define printf(A, ...) ;
#endif

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

    int size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double start, end;

    D_INT Grows = 7, Gcols = 9;
    D_INT ProcX = 2, ProcY = 3;
    D_INT blockX = 2, blockY = 2;

    void* mpicxt = BLACScxtInit('R', MPI_COMM_WORLD, ProcX, ProcY);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    void* Matrix = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (!my_rank)
        printf("Init D mat : %f\n", end - start);

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
    if (!my_rank)
        printf("Init set : %f\n", end - start);

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
        error = DMatSet(Matrix, ir, jc, (ir + 1.0) + I * (jc + 2.0));
        if (error)
        {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (!my_rank)
        printf("Setting mat : %f\n", end - start);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Process the Queue
    error = ProcessSetQueue(Matrix);

    end = MPI_Wtime();
    if (!my_rank)
        printf("SetQueue Process : %f\n", end - start);

    if (error)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /////print Local block
    // if (my_rank == 1)
    // {
    //     printf("******* Local block ******\n");
    //     print_loc(Matrix);
    // }

    // now time to get back the values
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    error = initiateGetQueue(Matrix, ele_this_cpu);

    end = MPI_Wtime();
    if (!my_rank)
        printf("GetQueue init : %f\n", end - start);

    if (error)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    D_Cmplx* received_buf = calloc(ele_this_cpu, sizeof(D_Cmplx));
    D_Cmplx* actual_buf = calloc(ele_this_cpu, sizeof(D_Cmplx));

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (D_LL_INT iele = ele_this_cpu - 1; iele >= 0; --iele)
    {
        D_LL_INT gele = iele + shift;
        D_LL_INT ir = gele / Gcols;
        D_LL_INT jc = gele % Gcols;
        actual_buf[iele] = (ir + 1.0) + I * (jc + 2.0);
        error = DMatGet(Matrix, ir, jc, received_buf + iele);
        if (error)
        {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    end = MPI_Wtime();
    if (!my_rank)
        printf("GetQueue Set : %f\n", end - start);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    ProcessGetQueue(Matrix);

    end = MPI_Wtime();
    if (!my_rank)
        printf("GetQueue Process : %f\n", end - start);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Check if both elements are true;
    bool check_mats = true;
    for (D_LL_INT iele = 0; iele < ele_this_cpu; ++iele)
    {
        if (cabs(actual_buf[iele] - received_buf[iele]) > 1e-7)
        {
            check_mats = false;
        }
    }

    bool reduce_check_res;
    MPI_Reduce(&check_mats, &reduce_check_res, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();
    if (!my_rank)
        printf("End Checking : %f\n", end - start);

#ifdef NOPRINT
#undef printf
#endif
    if (my_rank == 0)
    {
        if (reduce_check_res)
            printf("Test Passed :) \n");
        else
            printf("Test Failed :( \n");
    }

    free(received_buf);
    free(actual_buf);
    free_D_Matrix(Matrix);
    BLACScxtFree(mpicxt);

    MPI_Finalize();
    return 0;
}

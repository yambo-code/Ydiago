#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "../diago.h"
#include "../solvers.h"
#include "tests.h"

#undef NOPRINT

#ifdef NOPRINT
#define printf(A, ...) ;
#endif

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double start, end;

    D_INT Grows = 16, Gcols = 16;
    D_INT ProcX = 2, ProcY = 3;
    D_INT blockX = 2, blockY = 2;

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
        error = DMatSet(Matrix, ir, jc, (ir + 1.0) + I * (jc + 1.0));
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

    struct D_Matrix* matA = Matrix;
    D_LL_INT nloc_elem = matA->ldims[0] * matA->ldims[1];

    D_float* matA_out = calloc(nloc_elem + 1, sizeof(*matA_out));

    Construct_BSE_RealHam(Matrix, matA_out);

    D_float* WORK = calloc(2 * blockX * blockY, sizeof(*WORK));

    D_INT ICPRNT = 1;
    D_INT IA = 1, NOUT = 6;
    D_INT DESCA[9];

    set_descriptor(matA, DESCA);

    if (!my_rank)
    {
        struct D_Matrix* mat = matA;
        for (int i = 0; i < mat->ldims[0]; ++i)
        {
            for (int j = 0; j < mat->ldims[1]; ++j)
            {
                D_float tmp_d = matA_out[i * mat->lda[0] + j * mat->lda[1]];
                printf("%.1f,  ", tmp_d);
            }
            printf("\n");
        }
    }

    free(WORK);
    free(matA_out);
    free_D_Matrix(Matrix);
    BLACScxtFree(mpicxt);

    MPI_Finalize();
    return 0;
}

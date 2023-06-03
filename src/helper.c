#include "bse_diagonalize.h"

void Function(Real2Imag, Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) Complex_mat, EL_type(ElDistMatrix,Nd_floatS) Real_mat)
{
    /*
    INPUTS:
    Converts real to imag matrix
    */
    
    ElInt bse_dim, error; // bse_dim in TDA and error int

    MPI_Comm comm, comm2 ;

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (Complex_mat, &comm);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixDistComm,Nd_floatS) (Real_mat, &comm2);
    EL_ABORT_ON_ERROR( error );

    if (comm != comm2)
    {
        printf(" Real to imag conversion failed due to inconsistant MPI_COMM \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    ElInt lrows, lcols; // local block dims of the distributed matrix A

    /* Get the local block dims */
    error = EL_Function(ElDistMatrixLocalHeight,Nd_floatS) (Real_mat,&lrows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixLocalWidth,Nd_floatS)  (Real_mat,&lcols);
    EL_ABORT_ON_ERROR( error );


    for (ElInt r = 0; r<lrows; ++r)
    {   
        ElInt irow; // Global index of local row index r
        error = EL_Function(ElDistMatrixGlobalRow,Nd_floatS) (Real_mat, r, &irow ); 
        EL_ABORT_ON_ERROR( error );

        for (ElInt c = 0; c<lcols; ++c)
        {

            float element;
            float complex element_c;

            ElInt jcol; // Global index of local col index c 

            error = EL_Function(ElDistMatrixGlobalCol,Nd_floatS) (Real_mat, c, &jcol );
            EL_ABORT_ON_ERROR( error );

            error = EL_Function(ElDistMatrixGetLocal,Nd_floatS) (Real_mat,r,c,&element);
            EL_ABORT_ON_ERROR( error );

            element_c = element ; 

            error = EL_Function(ElDistMatrixSet,Nd_cmplxS) (Complex_mat,irow,jcol,element_c);
            EL_ABORT_ON_ERROR( error );
            
        }
    }
    MPI_Barrier(comm);

}




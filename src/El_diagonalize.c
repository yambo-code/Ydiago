#include "bse_diagonalize.h"


void Function(geev,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) A, EL_type(ElDistMatrix,Nd_cmplxS) W,
                            EL_type(ElDistMatrix,Nd_cmplxS) Z_left, EL_type(ElDistMatrix,Nd_cmplxS) Z_right)
{   
    /*
    INPUTS/OUTPUTS: A (Original matrix. In the end this will be upper triangular form)
                    W (Eigen values)
                    Z_left (Transpose of Left eigenvectors) 
                    Z_right (Right eigenvectors)
    */  

    ElError error;

    error = EL_Function(ElEigDist,Nd_cmplxS) (A, W, Z_right);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElCopyDist,Nd_cmplxS) (Z_right,Z_left);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElInverseDist,Nd_cmplxS) (Z_left); // Z_left = (Z_right^-1)^T but we are not transposing here
    EL_ABORT_ON_ERROR( error );

}

void Function(heev,Nd_cmplxS) (char uplo, EL_type(ElDistMatrix,Nd_cmplxS) A, EL_type(ElDistMatrix,Nd_floatS) W, \
                            EL_type(ElDistMatrix,Nd_cmplxS) Z, bool eig_vals_only, bool show_progress, \
                            ND_int * n_eig_idx_range, BS_float * eig_val_range)
{   
    /*
    INPUTS/OUTPUTS: A (Original matrix. In the end this will be upper triangular form)
                    W (Eigen values)
                    Z (Eigenvectors)
                    eig_vals_only --> If only eigenvalues are required
                    DC_algo -> Use Divide and conquor instead of MRRR(default)
                    n_eig_idx_range[2] = index range
                    eig_val_range[2]  = eigen value range
                    if both the above are NULL, entire spectrum is computed
    */
    ElError error;

    int rank;

    ElMPIWorldRank(&rank);

    bool partial_eig = false;

    EL_type(ElHermitianEigCtrl,Nd_cmplxS) ctrl;
    
    error = EL_Function(ElHermitianEigCtrlDefault,Nd_cmplxS) (&ctrl);
        
    EL_ABORT_ON_ERROR( error );

    ElInt grows, gcols; // global rows and cols of A

    /* Get the global block dims */
    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (A,&grows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixWidth,Nd_cmplxS)  (A,&gcols);
    EL_ABORT_ON_ERROR( error );

    if (grows != gcols)
    {
        if (rank == 0) printf("Error : Something wrong with Matrix. Matrix given to heev is not square. \n");
        if (rank == 0) MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    if ( n_eig_idx_range != NULL && eig_val_range != NULL )
    {   
        if (rank == 0) printf("Error : Provide either number range or value range and not both \n");
        if (rank == 0) MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    if ( n_eig_idx_range != NULL || eig_val_range != NULL ) partial_eig = true;

    EL_type(ElHermitianEigSubset,Nd_floatS) Eigsubset;

    if (n_eig_idx_range != NULL)
    {
        Eigsubset.indexSubset = true;
        Eigsubset.rangeSubset = false;

        if (n_eig_idx_range[0]>grows-1)
        {
            if (rank == 0) printf("!!!!!!! Warning : Provided number of eig values is greater \
            than dimension of the matrix. Setting the lower bound to 0 \n");
            n_eig_idx_range[0] = 0;
        }
        if (n_eig_idx_range[1]>grows-1)
        {
            if (rank == 0) printf("!!!!!!! Warning : Provided number of eig values is greater \
            than dimension of the matrix. Setting the upper bound to dimension of the matrix \n");
            n_eig_idx_range[1] = grows-1;
        }

        Eigsubset.lowerIndex  = n_eig_idx_range[0];
        Eigsubset.upperIndex  = n_eig_idx_range[1];
        
        Eigsubset.lowerBound  = 0; // random
        Eigsubset.upperBound  = 0; // random

    }

    else if (eig_val_range != NULL) 
    {
        Eigsubset.indexSubset = false;
        Eigsubset.rangeSubset = true;

        Eigsubset.lowerBound  = eig_val_range[0];
        Eigsubset.upperBound  = eig_val_range[1];

        Eigsubset.lowerIndex  = 0; // random
        Eigsubset.upperIndex  = 0; // random
    }

    ctrl.tridiagEigCtrl.subset = Eigsubset;
    ctrl.tridiagEigCtrl.progress = show_progress;
    ctrl.tridiagEigCtrl.wantEigVecs = !eig_vals_only ;

    if (uplo == 'L')
    {   
        error = EL_Function(ElHermitianEigPairControlDist,Nd_cmplxS) (EL_LOWER, A, W, Z,ctrl);
        
        EL_ABORT_ON_ERROR( error );
    }
    else if (uplo == 'U')
    {   
        error = EL_Function(ElHermitianEigPairControlDist,Nd_cmplxS) (EL_UPPER, A, W, Z,ctrl);
        
        EL_ABORT_ON_ERROR( error );
    }
    else
    {   
        if (rank == 0) printf("heev only takes L or U for uplo \n");
        if (rank == 0) MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

}


void Function(Pencilheev,Nd_cmplxS) (ElPencil pencil, char uplo, EL_type(ElDistMatrix,Nd_cmplxS) A, \
                                EL_type(ElDistMatrix,Nd_cmplxS) B, EL_type(ElDistMatrix,Nd_floatS) W, \
                                EL_type(ElDistMatrix,Nd_cmplxS) Z_right, bool eig_vals_only, bool show_progress, \
                                ND_int * n_eig_idx_range, BS_float * eig_val_range)
{   
    /*
    INPUTS/OUTPUTS: A (Hermitian) --> In the end it contains Transpose of Z_left (i.e A is replaced by transpose of left eigen vectors)
                    B (Hermitian and positve definite) --> Contains garbage in the end
                    W (Eigen values)
                    Z_right (Right eigenvectors)
                    n_eig_idx_range[2] = index range
                    eig_val_range[2]  = eigen value range
                    if both the above are NULL, entire spectrum is computed
    */
    ElError error;

    bool partial_eig = false;

    EL_type(ElHermitianEigCtrl,Nd_cmplxS) ctrl;
    
    error = EL_Function(ElHermitianEigCtrlDefault,Nd_cmplxS) (&ctrl);
        
    EL_ABORT_ON_ERROR( error );

    if ( n_eig_idx_range != NULL && eig_val_range != NULL )
    {   
        printf("Error : Provide either number range or value range and not both \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    if ( n_eig_idx_range != NULL || eig_val_range != NULL ) partial_eig = true;

    EL_type(ElHermitianEigSubset,Nd_floatS) Eigsubset;

    if (n_eig_idx_range != NULL)
    {
        Eigsubset.indexSubset = true;
        Eigsubset.rangeSubset = false;

        Eigsubset.lowerIndex  = n_eig_idx_range[0];
        Eigsubset.upperIndex  = n_eig_idx_range[1];

        Eigsubset.lowerBound  = 0; // random
        Eigsubset.upperBound  = 0; // random

    }

    else if (eig_val_range != NULL) 
    {
        Eigsubset.indexSubset = false;
        Eigsubset.rangeSubset = true;

        Eigsubset.lowerBound  = eig_val_range[0];
        Eigsubset.upperBound  = eig_val_range[1];

        Eigsubset.lowerIndex  = 0; // random
        Eigsubset.upperIndex  = 0; // random
    }

    ctrl.tridiagEigCtrl.subset = Eigsubset;
    ctrl.tridiagEigCtrl.progress = show_progress;
    ctrl.tridiagEigCtrl.wantEigVecs = !eig_vals_only ;

    if (uplo == 'L')
    {   
        error = EL_Function(ElHermitianGenDefEigPairControlDist,Nd_cmplxS) (pencil, EL_LOWER, A, B, W, Z_right,ctrl);
        
        EL_ABORT_ON_ERROR( error );
    }
    else if (uplo == 'U')
    {   
        error = EL_Function(ElHermitianGenDefEigPairControlDist,Nd_cmplxS) (pencil, EL_UPPER, A, B, W, Z_right,ctrl);
        
        EL_ABORT_ON_ERROR( error );
    }
    else
    {   
        printf("Pencelheev only takes L or U for uplo \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    /* Copy the right eigenvectors to A*/
    error = EL_Function(ElCopyDist,Nd_cmplxS) (Z_right,A);
    EL_ABORT_ON_ERROR( error );

    /* B contains Cholesky facotization L (lower) or U (upper) */
    if (!partial_eig)
    {   
        /* If Not partial, inverse will give the eigen vectors*/
        /* Get the left eigen vectors Z_left = Z_right^-T*/
        error = EL_Function(ElInverseDist,Nd_cmplxS) (A); // Z_left = (Z_right^-1)^T but we are not transposing here
        EL_ABORT_ON_ERROR( error );
    }
    else
    {
        /* Get Eigenvvalue from Right eigenvectors*/
        /*
        Pencil          Right                  Left (as Transposed)
        AXBX       L^-H * y ; U^-1 * y       (L * y)^H      ; (U^H  * y)^H  // Trmm
        ABX        L^-H * y ; U^-1 * y       (L * y)^H      ; (U^H  * y)^H  // Trmm
        BAX        L * y    ; U^H  * y       (L^-H * y)^H   ; (U^-1 * y)^H  // Trsm
        */

        ElUpperOrLower uplo;
        ElOrientation orientation;

        if (pencil == EL_BAX)
        {   
            if (uplo == 'L')
            {
                uplo = EL_LOWER ; 
                orientation = EL_ADJOINT;
            }
            else
            {
                // uplo == 'U' gives error above if anything else
                uplo = EL_UPPER ; 
                orientation = EL_NORMAL;

            }

            error = EL_Function(ElTrsmDist,Nd_cmplxS)(EL_LEFT, uplo, orientation, EL_NON_UNIT, 1.0f, B, A);
            EL_ABORT_ON_ERROR( error );
        }
        else // for ABX and AXBX
        {   
            if (uplo == 'L')
            {
                uplo = EL_LOWER ; 
                orientation = EL_NORMAL; 
            }
            else
            {
                // uplo == 'U' gives error above if anything else
                uplo = EL_UPPER ; 
                orientation = EL_ADJOINT;

            }

            error = EL_Function(ElTrmmDist,Nd_cmplxS)(EL_LEFT, uplo, orientation, EL_NON_UNIT, 1.0f, B, A);
            EL_ABORT_ON_ERROR( error );

        }
        
        /* Perform the Hermitian conjutate*/

        /* Conjugate*/
        error = EL_Function(ElConjugateDist,Nd_cmplxS)(A);
        EL_ABORT_ON_ERROR( error );

        /* Perform Transpose */
        error = EL_Function(ElTransposeDist,Nd_cmplxS)(A,B); // Now B has left eigen vectors
        EL_ABORT_ON_ERROR( error );
        
        /* Finally Copy B to A */
        error = EL_Function(ElCopyDist,Nd_cmplxS) (B,A); // Copy B to A
        EL_ABORT_ON_ERROR( error );
    }

}


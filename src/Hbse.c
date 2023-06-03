#include "bse_diagonalize.h"


static void Function(GetSymmetrizeQuery,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) block, char ulpo, ElInt * QueueNo);



/** Returns kernal with deltaE's added to diagonal elements **/
void Function(AddDeltaE2Diag,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) Kernal, BS_cmplx * delta_energies, \
                        BS_cmplx alpha, BS_cmplx beta, bool make_diag_real)
{
    /*
    INPUTS:
    Kernal --> kernal matrix elements
    delta_energies -> Delta E array of size (BSE_SIZE). 
    Kernal = beta* diag(Delta E) + alpha*Kernal
    // BS_overlap(i,j)=sum_k conjg(BS_V_left(k,i))*BS_V_right(k,j)
    !!!!!!! The Order of DeltaE must be same as bse dimesion
    */
    
    ElInt bse_dim, error; // bse_dim in TDA and error int

    MPI_Comm comm ;

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (Kernal, &comm);
    EL_ABORT_ON_ERROR( error );

    /* Get the global block dims */
    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (Kernal,&bse_dim);
    EL_ABORT_ON_ERROR( error );


    ElInt lrows, lcols; // local block dims of the distributed matrix A

    /* Get the local block dims */
    error = EL_Function(ElDistMatrixLocalHeight,Nd_cmplxS) (Kernal,&lrows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixLocalWidth,Nd_cmplxS)  (Kernal,&lcols);
    EL_ABORT_ON_ERROR( error );

    //scale the kernal to alpha
    error = EL_Function(ElScaleDist,Nd_cmplxS) (alpha, Kernal);
    EL_ABORT_ON_ERROR( error );

    for (ElInt r = 0; r<lrows; ++r)
    {   
        ElInt irow; // Global index of local row index r
        error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (Kernal, r, &irow ); 
        EL_ABORT_ON_ERROR( error );

        for (ElInt c = 0; c<lcols; ++c)
        {
            ElInt jcol; // Global index of local col index c 
            error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (Kernal, c, &jcol );
            EL_ABORT_ON_ERROR( error );

            if (jcol == irow)
            {
                // Read the particular element
                BS_cmplx element;

                error = EL_Function(ElDistMatrixGetLocal,Nd_cmplxS) (Kernal,r,c,&element);
                EL_ABORT_ON_ERROR( error );

                // Set the element
                if (make_diag_real) element = creal(element);
                
                element = element + beta*delta_energies[irow]; 

                error = EL_Function(ElDistMatrixSetLocal,Nd_cmplxS) (Kernal,r,c,element);
                EL_ABORT_ON_ERROR( error );
            
            }
            
        }
    }

    MPI_Barrier(comm);

}


/**** This loads the sub block buffers to kernal. *****/
void Function(load_full_kernal_block,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) Kernal, EL_type(ElDistMatrix,Nd_cmplxS) block,\
    ElInt shift_x, ElInt shift_y, bool transpose, bool conjugate, BS_cmplx alpha, BS_cmplx beta)
{   
    /*

    This loads the sub block buffers to kernal.

    INPUTS:

    * Kernal --> Total kernal matrix to fill

    *block to be loaded from (shift_x,shift_y) point (i.e buffer block) i.e Kernal[shift_x:,shift_y:] = alpha * OP(block) + beta
    OP(block) = conj(block) if conjugate is true else OP(block) is block

    * block_type ('H' or 'S' or 'K') --> IF the block is hermitian give 'H' , If the block is symmetric give 'S' and 'SK' for skew symmetric. 
    anything other, ulpo will be discarded

    * ulpo ('L; or 'U')--> In case the block is filled only upper ('U') or lower ('L'), using block_type, the other half is filled.

        | R   C |
    K = |       |
        |-C* -R*|
    
    */
    

    ElInt bse_dim, block_dim, error; // Kernal dimension, block dimensiton( = Kernal dimension/2) and error int

    MPI_Comm comm, comm_temp ;

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (Kernal, &comm);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (block, &comm_temp);
    EL_ABORT_ON_ERROR( error );

    if (comm_temp != comm)
    {
        printf(" Kernal and sub block must be on same MPI world  \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    /* Get the global block dims */
    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (Kernal,&bse_dim);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (block,&block_dim);
    EL_ABORT_ON_ERROR( error );

    if (bse_dim/2 != block_dim)
    {
        printf(" Block Dimension must be half of kernal \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    ElInt lrows, lcols; // local block dims of the distributed matrix 

    /* Get the local block dims */
    error = EL_Function(ElDistMatrixLocalHeight,Nd_cmplxS) (block,&lrows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixLocalWidth,Nd_cmplxS)  (block,&lcols);
    EL_ABORT_ON_ERROR( error );

    EL_Function(ElDistMatrixReserve,Nd_cmplxS) (Kernal, lrows*lcols);

    for (ElInt r = 0; r<lrows; ++r)
    {   
        ElInt irow; // Global index of local row index r for block 
        error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
        EL_ABORT_ON_ERROR( error );

        for (ElInt c = 0; c<lcols; ++c)
        {
            ElInt jcol; // Global index of local col index c for block 
            error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
            EL_ABORT_ON_ERROR( error );

            BS_cmplx element;

            error = EL_Function(ElDistMatrixGetLocal,Nd_cmplxS) (block,r,c,&element);
            EL_ABORT_ON_ERROR( error );

            // Set the element

            if (conjugate) element = conj(element);

            element = alpha*element + beta;

            //printf("%f + %fI \n",creal(element),cimag(element));

            if (transpose) error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (Kernal, jcol+shift_x,irow+shift_y, element);
            else           error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (Kernal, irow+shift_x,jcol+shift_y, element);
            EL_ABORT_ON_ERROR( error ); 
            
        }
    }
    EL_Function(ElDistMatrixProcessQueues,Nd_cmplxS) (Kernal);

    MPI_Barrier(comm);

}

static void Function(GetSymmetrizeQuery,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) block, char ulpo, ElInt * QueueNo)
{   
    /*

    This Function gives Queue Number need for Symmetrize Function

    * ulpo ('L; or 'U')--> In case the block is filled only upper ('U') or lower ('L'), using block_type, the other half is filled.*/
    ElInt error; // bse_dim in TDA and error int

    MPI_Comm comm ;

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (block, &comm);
    EL_ABORT_ON_ERROR( error );

    ElInt lrows, lcols; // local block dims of the distributed matrix 

    /* Get the local block dims */
    error = EL_Function(ElDistMatrixLocalHeight,Nd_cmplxS) (block,&lrows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixLocalWidth,Nd_cmplxS)  (block,&lcols);
    EL_ABORT_ON_ERROR( error );

    *QueueNo = 0;

    if (ulpo == 'L')
    {
        for (ElInt r = 0; r<lrows; ++r)
        {   
            ElInt irow; // Global index of local row index r for block 
            error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
            EL_ABORT_ON_ERROR( error );

            for (ElInt c = 0; c<lcols; ++c)
            {
                ElInt jcol; // Global index of local col index c for block 
                error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
                EL_ABORT_ON_ERROR( error );

                if (irow >= jcol) *QueueNo = *QueueNo + 1; // we are not including the diagonal elements as they are not altered
            
            }
        }
        return ;
    }

    else if (ulpo == 'U')
    {
        for (ElInt r = 0; r<lrows; ++r)
        {   
            ElInt irow; // Global index of local row index r for block 
            error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
            EL_ABORT_ON_ERROR( error );

            for (ElInt c = 0; c<lcols; ++c)
            {
                ElInt jcol; // Global index of local col index c for block 
                error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
                EL_ABORT_ON_ERROR( error );

                if (jcol >= irow) *QueueNo = *QueueNo + 1; // we are not including the diagonal elements as they are not altered
            }
        }
        return ;
    }

    else
    {   
        *QueueNo = 0;
        return ;
    }

}


void Function(Symmetrize,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) block, char ulpo, char block_type)
{   
    /*

    This Function gives Queue Number need for Symmetrize Function

    * ulpo ('L; or 'U')--> In case the block is filled only upper ('U') or lower ('L'), using block_type, the other half is filled.

    * block_type ('H' or 'S' or 'K') --> IF the block is hermitian give 'H' , If the block is symmetric give 'S' and 'SK' for skew symmetric. 
    anything other, ulpo will be discarded */
    ElInt error; // bse_dim in TDA and error int

    MPI_Comm comm ;

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (block, &comm);
    EL_ABORT_ON_ERROR( error );

    if (ulpo == 'L' || ulpo == 'U')
    {
        if(block_type != 'H' && block_type != 'S' && block_type != 'K')
        {
            printf(" When using ulpo, the matrix must be either symmetric or hermitian or skew symmetric  \n");
            MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
        }
    }
    else return ;
    

    ElInt bse_dim, block_dim; // Kernal dimension, block dimensiton( = Kernal dimension/2) and error int

    /* Get the global block dims */

    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (block,&block_dim);
    EL_ABORT_ON_ERROR( error );

    ElInt lrows, lcols; // local block dims of the distributed matrix 

    /* Get the local block dims */
    error = EL_Function(ElDistMatrixLocalHeight,Nd_cmplxS) (block,&lrows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixLocalWidth,Nd_cmplxS)  (block,&lcols);
    EL_ABORT_ON_ERROR( error );

    ElInt QueueNo;

    Function(GetSymmetrizeQuery,Nd_cmplxS) (block, ulpo, &QueueNo);
    
    if (QueueNo <1) return ;

    BS_cmplx * buffer = malloc(sizeof(BS_cmplx)*QueueNo);

    if (buffer == NULL)
    {
        printf(" Failed to create Buffer array in Symmetrize Function  \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }    

    if (ulpo == 'L')
    {   
        ElInt buffer_Count = 0;
        for (ElInt r = 0; r<lrows; ++r)
        {   
            ElInt irow; // Global index of local row index r for block 
            error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
            EL_ABORT_ON_ERROR( error );

            for (ElInt c = 0; c<lcols; ++c)
            {
                ElInt jcol; // Global index of local col index c for block 
                error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
                EL_ABORT_ON_ERROR( error );

                if (irow >= jcol)
                {
                    error = EL_Function(ElDistMatrixGetLocal,Nd_cmplxS) (block,r,c,buffer+buffer_Count);
                    EL_ABORT_ON_ERROR( error );

                    buffer_Count++ ;

                }
            
            }
        }
    }

    else if (ulpo == 'U')
    {   
        ElInt buffer_Count = 0;
        for (ElInt r = 0; r<lrows; ++r)
        {   
            ElInt irow; // Global index of local row index r for block 
            error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
            EL_ABORT_ON_ERROR( error );

            for (ElInt c = 0; c<lcols; ++c)
            {
                ElInt jcol; // Global index of local col index c for block 
                error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
                EL_ABORT_ON_ERROR( error );

                if (jcol >= irow)
                {
                    error = EL_Function(ElDistMatrixGetLocal,Nd_cmplxS) (block,r,c,buffer+buffer_Count);
                    EL_ABORT_ON_ERROR( error );

                    buffer_Count++ ;

                }
            
            }
        }
    }

    error = EL_Function(ElZeroDist,Nd_cmplxS)  (block);
    EL_ABORT_ON_ERROR( error );

    if (ulpo == 'L')
    {   
        ElInt buffer_Count = 0;
        EL_Function(ElDistMatrixReserve,Nd_cmplxS) (block, QueueNo);

        for (ElInt r = 0; r<lrows; ++r)
        {   
            ElInt irow; // Global index of local row index r for block 
            error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
            EL_ABORT_ON_ERROR( error );

            for (ElInt c = 0; c<lcols; ++c)
            {
                ElInt jcol; // Global index of local col index c for block 
                error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
                EL_ABORT_ON_ERROR( error );

                if (irow >= jcol)
                {
                    // Read the particular element
                    BS_cmplx element = buffer[buffer_Count];
                    // Set the element
                    if      (block_type == 'H') element = conj(element);
                    else if (block_type == 'K') element = -element;

                    // printf("%f + %fI \n",creal(element),cimag(element));

                    error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (block, irow,jcol, buffer[buffer_Count]);
                    EL_ABORT_ON_ERROR( error );

                    if(irow != jcol)
                    {
                        error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (block, jcol, irow, element);
                        EL_ABORT_ON_ERROR( error );
                    }
                    buffer_Count++;
                }
            
            }
        }
        EL_Function(ElDistMatrixProcessQueues,Nd_cmplxS) (block);
        
    }

    else if (ulpo == 'U')
    {   
        ElInt buffer_Count = 0;
        EL_Function(ElDistMatrixReserve,Nd_cmplxS) (block, QueueNo);

        for (ElInt r = 0; r<lrows; ++r)
        {   
            ElInt irow; // Global index of local row index r for block 
            error = EL_Function(ElDistMatrixGlobalRow,Nd_cmplxS) (block, r, &irow ); 
            EL_ABORT_ON_ERROR( error );

            for (ElInt c = 0; c<lcols; ++c)
            {
                ElInt jcol; // Global index of local col index c for block 
                error = EL_Function(ElDistMatrixGlobalCol,Nd_cmplxS) (block, c, &jcol );
                EL_ABORT_ON_ERROR( error );

                if (jcol >= irow)
                {
                    // Read the particular element
                    BS_cmplx element = buffer[buffer_Count];
                    // Set the element
                    if      (block_type == 'H') element = conj(element);
                    else if (block_type == 'K') element = -element;

                    // printf("%f + %fI \n",creal(element),cimag(element));

                    error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (block,irow,jcol,buffer[buffer_Count]);
                    EL_ABORT_ON_ERROR( error );

                    if(irow != jcol)
                    {
                        error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (block, jcol, irow, element);
                        EL_ABORT_ON_ERROR( error );
                    }
                    buffer_Count++;
                }
            
            }
        }
        EL_Function(ElDistMatrixProcessQueues,Nd_cmplxS) (block);
        
    }

    free(buffer);
    MPI_Barrier(comm);

}



    





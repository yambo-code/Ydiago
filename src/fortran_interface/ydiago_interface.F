module YDIAGO_interface
    use, intrinsic :: iso_c_binding
    implicit none
  
    ! Define Fortran types corresponding to C types
    ! These types are define in diago.h
#if defined(WITH_DOUBLE) || defined(_DOUBLE)
    integer, parameter :: YDIAGO_FLOAT  = C_DOUBLE
    integer, parameter :: YDIAGO_CMPLX  = C_DOUBLE_COMPLEX
#else
    integer, parameter :: YDIAGO_FLOAT  = C_FLOAT
    integer, parameter :: YDIAGO_CMPLX  = C_FLOAT_COMPLEX
#endif
    integer, parameter :: YDIAGO_INT    = C_INT 
    integer, parameter :: YDIAGO_LL_INT = C_LONG_LONG
    integer, parameter :: ERROR_INT     = C_INT 
    ! This is D_INT in diago.h. This must be set to the same.
    
    ! Define the interface for the C functions
    interface
  
      ! Function: Heev
      integer(ERROR_INT) function Heev(DmatA, ulpo, neigs_range, eigval_range, eig_vals, Deig_vecs, neig_found) bind(C, name="Heev")
        
      import :: c_ptr, c_char, YDIAGO_INT, ERROR_INT
        
        type(c_ptr), value :: DmatA 
        ! distributed matrix which we want to diagonalize
        
        character(c_char), value :: ulpo
        ! 'U' if upper triangule is set, 'L' is lower. We only need to set either upper or lower
        
        type(c_ptr), value :: neigs_range 
        ! c pointer to YDIAGO_INT  type (target) array i.e c_loc(neigs_range(2)) or pass c_null_ptr
        ! index range of eigenvalues & vectors required. for Ex: if we need 10-100 eigenvlaues then is [10,100]
        
        type(c_ptr), value :: eigval_range !c_loc(eigval_range(2)) or pass c_null_ptr
        ! c pointer to YDIAGO_FLOAT  type (target) array i.e c_loc(eigval_range(2)) or pass c_null_ptr
        ! range of eigenvalues & vectors required. (use either neigs_range or eigval_range and pass c_null_ptr)
        ! passing c_null_ptr for both neigs_range and eigval_range will simply compute all eigenvalues and eigenvectors.
        
        type(c_ptr), value :: eig_vals
        ! Eigenvalues array of dimesion same as size of matrix (Enough though only subset is wanted).
        ! c_loc(eig_vals(2)) Type (YDIAGO_CMPLX) 
        ! Note :Every process gets the full set
        
        type(c_ptr), value :: Deig_vecs
        ! Distributed matrix for eigenvectors. must have same dimension as DmatA
        
        integer(YDIAGO_INT) :: neig_found
        ! On output, this is set to number of eigenvalues found.
      end function Heev
  
      ! Function: BSE_Solver
      integer(ERROR_INT) function BSE_Solver(DmatA, neigs_range, eigval_range, &
      &             eig_vals, Deig_vecs, neigs_found) bind(C, name="BSE_Solver")
        
        import :: c_ptr, YDIAGO_INT, ERROR_INT
        
        type(c_ptr), value :: DmatA
        ! distributed matrix which we want to diagonalize. The entire matrix must be filled !
        ! ! (2n, 2n) where n is size of resonant block
        
        type(c_ptr), value :: neigs_range 
        ! c pointer to YDIAGO_INT  type (target) array i.e c_loc(neigs_range(2)) or pass c_null_ptr
        ! index range of eigenvalues & vectors required. for Ex: if we need 10-100 eigenvlaues then is [10,100]

        type(c_ptr), value :: eigval_range 
        ! c pointer to YDIAGO_FLOAT  type (target) array i.e c_loc(eigval_range(2)) or pass c_null_ptr
        ! range of eigenvalues & vectors required. (use either neigs_range or eigval_range and pass c_null_ptr)
        ! passing c_null_ptr for both neigs_range and eigval_range will simply compute all eigenvalues and eigenvectors.
        
        type(c_ptr), value :: eig_vals
        ! Eigenvalues array of dimesion same as size of matrix (Enough though only subset is wanted).
        ! c_loc(eig_vals(2)) Type (YDIAGO_CMPLX)
        ! Note :Every process gets the full set
        
        type(c_ptr), value :: Deig_vecs
        ! Distributed matrix for eigenvectors. must have same dimension as DmatA. Only eigen vectors for postive eigen values are returned
        ! You can generate eigenvalues and left/right eigenvectors of correspinding negative counterparts
        !     Eigen values come in pair i.e (-lambda, lambda).
        !     Only computes postive eigenvalues and their correspoing right
        !     eigen vectors. From this we can retreive left eigen vectors
        !     for positive eigen values, and left and right eigenvectors of negative
        !     eigen values
    
        !         Right eigenvectors         Left eigenvectors
        !           +ve     -ve               +ve       -ve
        !     X = [ X_1, conj(X_2) ]    Y = [  X_1, -conj(X_2)]
        ! *        [X_2, conj(X_1) ]        [ -X_2,  conj(X_1)],
    
        !     The code returns [X1,X2] as eigen vectors
        integer(YDIAGO_INT) :: neigs_found
        ! On output, this is set to number of eigenvalues found.
      end function BSE_Solver
  
      ! Function: Geev
      integer(ERROR_INT) function Geev(DmatA, eig_vals, Deig_vecsL, Deig_vecsR) bind(C, name="Geev")
      ! Note for now donot use this function, due to numerical unstability of some of scalapack functions
      ! the errors comeout to be are large. This is just a ref implementation and can be changed in future 
      ! if they improve scalapack functions !
      import :: c_ptr, YDIAGO_INT, ERROR_INT

        type(c_ptr), value :: DmatA
        ! distributed matrix which we want to diagonalize. The entire matrix must be filled !
        
        type(c_ptr), value ::  eig_vals
        ! Eigenvalues array of dimesion same as size of matrix (Enough though only subset is wanted).
        ! c_loc(eig_vals(2)) Type (YDIAGO_CMPLX)
        ! Note :Every process gets the full set

        type(c_ptr), value :: Deig_vecsL
        ! Distributed matrix for left eigenvectors. (pass c_null_ptr if not required)

        type(c_ptr), value :: Deig_vecsR
        ! Distributed matrix for right eigenvectors. (pass c_null_ptr if not required)
      end function Geev

#if defined(_ELPA) || defined(WITH_ELPA)
      ! Function: Heev_Elpa
      integer(ERROR_INT) function Heev_Elpa(D_mat, eig_vals, Deig_vecs, neigs, elpa_solver, gpu_type, nthreads) bind(C, name="Heev_Elpa")

        import :: c_ptr, c_char, YDIAGO_INT, ERROR_INT
      
        type(c_ptr), value :: D_mat
        ! distributed matrix which we want to diagonalize. The entire matrix must be filled !
        
        type(c_ptr), value ::  eig_vals
        ! Eigenvalues array of dimesion same as size of matrix (Enough though only subset is wanted).
        ! c_loc(eig_vals(2)) Type (YDIAGO_CMPLX)
        ! Note :Every process gets the full set

        type(c_ptr), value :: Deig_vecs
        ! Distributed matrix for eigenvectors. (pass c_null_ptr if not required)

        integer(YDIAGO_INT), value :: neigs
        ! number of eigenvalues required. if neigs < diminsion of matrix, then first neigs eigenvals are computed

        integer(YDIAGO_INT), value :: elpa_solver
        ! WHich Solver to use ELPA-1 or ELPA-2 (2 is generally faster on CPUs and scales good)
        ! Refer Elpa documentation for more details
        
        type(c_ptr), value :: gpu_type
        ! If compiled with GPU, provide which type of gpu is use. only reference when compiled with gpu support
        ! Here are the values : 
        ! c_loc('nvidia-gpu'//c_null_char) for Nvidia gpus
        ! c_loc('amd-gpu'//c_null_char)    for amd gpus
        ! c_loc('intel-gpu'//c_null_char) for intel-gpus
        ! Pass c_null_ptr if donot want to run on gpus

        integer(YDIAGO_INT), value :: nthreads
        ! Set number of threads, only reference when compiled with openmp support
        !
      end function Heev_Elpa
  
      ! Function: BSE_Solver_Elpa
      integer(ERROR_INT) function BSE_Solver_Elpa(D_mat, eig_vals, Deig_vecs, elpa_solver, gpu_type, nthreads) bind(C, name="BSE_Solver_Elpa")
        ! This will diagonalize full hamilitioan (unlike scalapack version)
        import :: c_ptr, c_char, YDIAGO_INT, ERROR_INT
        
        type(c_ptr), value :: D_mat
        ! distributed matrix which we want to diagonalize. The entire matrix must be filled !
        ! ! (2n, 2n) where n is size of resonant block

        type(c_ptr), value ::  eig_vals
        ! Eigenvalues array of dimesion same as size of matrix (Enough though only subset is wanted).
        ! c_loc(eig_vals(2)) Type (YDIAGO_CMPLX)
        ! Note :Every process gets the full set

        type(c_ptr), value :: Deig_vecs
        ! Distributed matrix for eigenvectors. 

        integer(YDIAGO_INT), value :: elpa_solver
        ! WHich Solver to use ELPA-1 or ELPA-2 (2 is generally faster on CPUs and scales good)
        ! Refer Elpa documentation for more details

        type(c_ptr), value :: gpu_type
        ! If compiled with GPU, provide which type of gpu is use. only reference when compiled with gpu support
        ! Here are the values : 
        ! c_loc('nvidia-gpu'//c_null_char) for Nvidia gpus
        ! c_loc('amd-gpu'//c_null_char)    for amd gpus
        ! c_loc('intel-gpu'//c_null_char) for intel-gpus
        ! Pass c_null_ptr if donot want to run on gpus

        integer(YDIAGO_INT), value :: nthreads
        ! Set number of threads, only reference when compiled with openmp support

      end function BSE_Solver_Elpa
#endif

      ! Function: BLACScxtInit
      type(c_ptr) function BLACScxtInit_Fortran(layout, comm, ProcX, ProcY) bind(C, name="BLACScxtInit_Fortran")
        ! returns a opaquie mpicontxt which has blacs grid info. 
        ! must be freed in the end with BLACScxtFree function
        import :: c_ptr, c_char, YDIAGO_INT 
        
        character(c_char), value :: layout
        ! 'C' or 'R', Layout of blacs grid. Really does not matter. chose anything.    
        
        integer(YDIAGO_INT), value :: comm
        ! Comm interger of MPI communicator
        
        integer(YDIAGO_INT), value :: ProcX
        ! Number of processor rows 
        
        integer(YDIAGO_INT), value :: ProcY
        ! Number of processor cols
    end function BLACScxtInit_Fortran
  
      ! Function: BLACScxtFree
      subroutine BLACScxtFree(mpicontxt) bind(C, name="BLACScxtFree")
        ! Frees the BLACScxt created with BLACScxtInit_Fortran.
        import :: c_ptr
        type(c_ptr), value :: mpicontxt
      end subroutine BLACScxtFree
  
      ! Function: init_D_Matrix
      type(c_ptr) function init_D_Matrix(Grows, Gcols, blockX, blockY, mpicontxt) bind(C, name="init_D_Matrix")
        ! Initiates a Distributed 2D block cyclic matrix.
        import :: c_ptr, YDIAGO_INT, ERROR_INT
        
        integer(YDIAGO_INT), value :: Grows
        ! Global number of rows of a Matrix
        
        integer(YDIAGO_INT), value :: Gcols
        ! Global number of cols of a Matrix

        integer(YDIAGO_INT), value :: blockX
        ! Block size along the rows
        
        integer(YDIAGO_INT), value :: blockY
        ! Block size along the cols

        type(c_ptr), value :: mpicontxt
        !
      end function init_D_Matrix
  
      ! Function: free_D_Matrix
      subroutine free_D_Matrix(D_mat) bind(C, name="free_D_Matrix")
        import :: c_ptr
        type(c_ptr), value :: D_mat
      end subroutine free_D_Matrix
  
      ! Function: set_descriptor
      integer(ERROR_INT) function set_descriptor(D_mat, desc) bind(C, name="set_descriptor")
        ! Set blacs desciptor
        import :: c_ptr, YDIAGO_INT, ERROR_INT
        !
        type(c_ptr), value :: D_mat 
        !Distributed matrix
        integer(YDIAGO_INT) :: desc(9)
      end function set_descriptor
  
      ! Function: initiateGetQueue
      integer(ERROR_INT) function initiateGetQueue(D_mat, nelements) bind(C, name="initiateGetQueue")
        ! Initate a queue to Get the indiviual elements of a Distributed matrix
        import :: c_ptr, YDIAGO_INT, ERROR_INT, YDIAGO_LL_INT

        type(c_ptr), value :: D_mat
        ! Distributed matrix

        integer(YDIAGO_LL_INT), value :: nelements
        ! Total number of elements each process is going to Get (does not have to same on all processes)
      end function initiateGetQueue
  
      ! Function: DMatGet
      integer(ERROR_INT) function DMatGet_fortran(D_mat, i, j, value1) bind(C, name="DMatGet_fortran")
        ! Get the elements of (i,j)
        import :: c_ptr, YDIAGO_INT, ERROR_INT
        !
        type(c_ptr), value :: D_mat
        !
        integer(YDIAGO_INT), value :: i
        !
        integer(YDIAGO_INT), value :: j
        !
        type(c_ptr), value :: value1
        ! Pointer to YDiago_cmplx c_loc(A(i,j))
        ! NM : FIX ME: Not sure we can pass A(i,j), gfortran passes address
        ! but is it compiler dependent? For now stick to hardway
        !complex(YDIAGO_CMPLX) :: value1
      end function DMatGet_fortran
  
      ! Function: ProcessGetQueue
      integer(ERROR_INT) function ProcessGetQueue(D_mat) bind(C, name="ProcessGetQueue")
        ! Once every element index is given, call this function to assign the values 
        ! to the given buffer.
        import :: c_ptr, ERROR_INT
        !
        type(c_ptr), value :: D_mat
      end function ProcessGetQueue
  
      ! Function: initiateSetQueue
      integer(ERROR_INT) function initiateSetQueue(D_mat, nelements) bind(C, name="initiateSetQueue")
        ! Initate a queue to Set the indiviual elements of a Distributed matrix
        import :: c_ptr, YDIAGO_INT, ERROR_INT, YDIAGO_LL_INT
        !
        type(c_ptr), value :: D_mat
        !
        integer(YDIAGO_LL_INT), value :: nelements
        ! Total number of elements each process is going to Set (does not have to same on all processes)
      end function initiateSetQueue
  
      ! Function: DMatSet
      integer(ERROR_INT) function DMatSet_fortran(D_mat, i, j, value1) bind(C, name="DMatSet_fortran")
        ! Set the (i,j) element of distributed matrix with value1
        import :: c_ptr, YDIAGO_INT, YDIAGO_CMPLX, ERROR_INT
        !
        type(c_ptr), value :: D_mat
        !
        integer(YDIAGO_INT), value :: i
        !
        integer(YDIAGO_INT), value :: j
        !
        complex(YDIAGO_CMPLX), value :: value1
      end function DMatSet_fortran
  
      ! Function: ProcessSetQueue
      integer(ERROR_INT) function ProcessSetQueue(D_mat) bind(C, name="ProcessSetQueue")
        ! Once every element index is set, call this function to assign the values 
        ! to the distributed matrix.
        import :: c_ptr, ERROR_INT
        !
        type(c_ptr), value :: D_mat
      end function ProcessSetQueue
  
    end interface
  
end module YDIAGO_interface
  

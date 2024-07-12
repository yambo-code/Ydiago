program main
  use mpi
  use iso_c_binding
  use YDIAGO_interface
  implicit none

  integer :: ierr, size, my_rank
  integer(YDIAGO_INT) :: Grows, Gcols, ProcX, ProcY, blockX, blockY
  type(c_ptr) :: mpicxt, Matrix, evecs
  real(8) :: start, end
  integer(YDIAGO_INT) :: ele_this_cpu, ele_rem, shift, gele, ir, jc
  complex(YDIAGO_FLOAT) :: tmp11
  integer(YDIAGO_INT) :: neig_found
  complex(YDIAGO_CMPLX), target, allocatable :: eig_vals(:)
  complex(YDIAGO_CMPLX), target, allocatable :: eig_vectors(:,:)
  integer(YDIAGO_INT) :: i, error
  integer(YDIAGO_INT) :: desc(9)

  call MPI_Init(ierr)

  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  IF (size .LT. 4) THEN
    WRITE(*,'(A)') 'At least 4 cpus are required to run this program'
    CALL MPI_Abort(MPI_COMM_WORLD, -1, ierr)
  END IF
  call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)

  Grows  = 16
  Gcols  = 16
  ProcX  = 2
  ProcY  = 2
  blockX = 8
  blockY = 8
  
  ! Always check the return errors, The library almost never exits 
  !(unless it is a fatal error) but gracefull exits. 
  !This means, the functions might fail without quiting the program.

  ! First we need to create a blacs grid 
  mpicxt = BLACScxtInit_Fortran('R', MPI_COMM_WORLD, ProcX, ProcY)
  ! Here I used rowmajor layout, one can also use 'C' which is
  ! Coloumn major. This is matter of taste.

  ! Let us create a block cyclic matrix which we want to diagonalize in memory
  Matrix = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt)
  ! Check is Matrix is properly initiated
  if (.not. c_associated(Matrix)) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  
  ! Similarly Create a distrbuted matrix for eigenvectors. 
  ! Note the code only returns Right eigenvectors even incase of non-TDA
  evecs = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt)
  ! check if evecs are initiated properly
  ! Eigenvalues are stored in columns of the distributed matrix i.e 
  ! evecs[:,i] represents the ith eigenvector


  if (.not. c_associated(evecs)) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  error = 0

  !! Not so important. Just seting up few things 
  ele_this_cpu = (Grows * Gcols) / size
  ele_rem = mod(Grows * Gcols, size)
  shift = ele_this_cpu * my_rank

  if (my_rank < ele_rem) then
    shift = shift + my_rank
    ele_this_cpu = ele_this_cpu + 1
  else
    shift = shift + ele_rem
  end if
  
  !! Now Let us now fill the matrix with random values. 
  !! each CPU has different elements in the and wants to set
  ! the distributed matrix (Matrix)
  ! Yambo calls this as Matrix folding

  ! Initiate the folding procedure (of element setup process)
  error = initiateSetQueue(Matrix, INT(ele_this_cpu,kind=YDIAGO_LL_INT))
  ! each cpus is setting ele_this_cpu number of elements to the
  ! Distributed matrix.
  if (error /= 0) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  do i = 1, ele_this_cpu
    gele = i + shift-1
    ir = gele / Gcols  + 1
    jc = mod(gele, Gcols) + 1
    tmp11 = csin(gele + cmplx(0.0_YDIAGO_FLOAT, cos(real(gele))))
    gele = ir + jc * Gcols
    tmp11 = tmp11 + conjg(csin(gele + cmplx(0.0_YDIAGO_FLOAT, cos(real(gele)))))
    
    !! Note set the (ir,jc) element to tmp11
    error = DMatSet_Fortran(Matrix, ir, jc, tmp11)

    !! Check the error
    if (error /= 0) then
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
  end do

  ! Once setting up the elements is finished, we finilize the setting procedure
  ! by call the processSetQueue function
  error = ProcessSetQueue(Matrix)
  ! After this, we sucessfull set the elements of the distributed matrix.
  ! Now we are good to go for diagonalization procedure.

  ! Check the error
  if (error /= 0) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  ! Now allocate the eigenvalues.
  allocate(eig_vals(Grows))
  eig_vals = 1
  ! Check if sucessfull allocated
  if (.not. allocated(eig_vals)) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  if (my_rank == 0) then
    print *, "Diagonalization......."
  end if

  !! Ignore. testing descriptor function
  error = set_descriptor(Matrix, desc)
  if (my_rank == 0) then
    do i = 1, 9 
      print *, desc(i) 
    end do
  end if


  ! Note call the solver you need. Here we are calling the hermitian solver
  ! This is overwritten by the number of eigenvalues found (only for scalapack)
  neig_found = 0
  
  ! Call the diagonalization funtion 
  
  ! Scalapack TDA 
  error = Heev(Matrix, 'L', c_null_ptr, c_null_ptr, c_loc(eig_vals), evecs, neig_found)
  !! Elpa :
  ! error = Heev_Elpa(Matrix, eig_vals, evecs, Grows, 2, c_null_ptr, 1);
  !!!!!!
  !! incase it is standard non-TDA-
  ! error = BSE_Solver(Matrix, c_null_ptr, c_null_ptr, eig_vals, evecs, neigs_found)
  ! Elpa:
  ! error = BSE_Solver_Elpa(Matrix, eig_vals, evecs, eig_vals, evecs, neigs_found)

  ! The first neig_found entries of the eig_vals are filled with the eigenvalues

  if (my_rank == 0) then
    do i = 1, neig_found 
      print *, real(eig_vals(i)), aimag(eig_vals(i))
    end do
  end if

  ! Free the initial matrix to freeup some space for eigen-vectors
  call free_D_Matrix(Matrix)

  ! Eigenvalues are stored in columns of the distributed matrix i.e 
  ! evecs[:,i] represents the ith eigenvector
  
  ! Now we Retrieve the components of the eigenvectors in which ever distrbutation you 
  ! want. For example I want to distribute the number of eigenvectors such that 
  ! each cpu has subset of full vectors i.e [:,neig_found/ncpus].
  ! This can be done by the get Queue
  
  
  ele_this_cpu = neig_found/size !CEILING(real(neig_found)/real(ncpus)) 
  shift  = ele_this_cpu * my_rank
  ele_rem = mod(neig_found, size)
  if (my_rank < ele_rem) then
    ele_this_cpu = ele_this_cpu + 1
    shift = shift + my_rank
  else 
    shift = shift + ele_rem
  endif
  
  allocate(eig_vectors(Grows,ele_this_cpu))
  
  eig_vectors = 0

  ! // Now lets us start the retrvel procedure

  ! First and foremost initiate the getQueue
  error = initiateGetQueue(evecs, INT(ele_this_cpu*grows,kind=YDIAGO_LL_INT))

  ! Check the error
  if (error /= 0) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  
  do jc = shift+1, shift+ele_this_cpu
    do ir = 1, grows
      error = dmatget_fortran(evecs, ir,jc, c_loc(eig_vectors(ir,jc-shift)))
      if (error /= 0) then
        print *, error
        call mpi_abort(mpi_comm_world, 1, ierr)
      end if
    enddo
  enddo
  
  ! Finalize the get queue
  error = ProcessGetQueue(evecs)
  if (error /= 0) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  
  ! if (my_rank == 0) then
  !   do jc = shift+1, shift+ele_this_cpu
  !     do ir = 1, grows
  !       print *, eig_vectors(ir,jc-shift)
  !     enddo
  !   enddo
  ! endif 

  
  if (my_rank == 0) then
    print *, 'Printing Norms'
    do jc = shift+1, shift+ele_this_cpu
        print *, sqrt( sum(abs(eig_vectors(:,jc-shift)**2 )))
    enddo
  endif 


  ! Free all the matrices 
  call free_D_Matrix(evecs)
  call BLACScxtFree(mpicxt)

  deallocate(eig_vectors)
  deallocate(eig_vals)

  call MPI_Finalize(ierr)

end program main


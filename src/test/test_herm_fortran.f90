program main
  use mpi
  use iso_c_binding
  use YDIAGO_interface
  implicit none

  integer :: ierr, size, my_rank
  integer(c_int) :: Grows, Gcols, ProcX, ProcY, blockX, blockY
  type(c_ptr) :: mpicxt, Matrix, Matrix_Z
  real(8) :: start, end
  integer(c_int) :: ele_this_cpu, ele_rem, shift, gele, ir, jc
  integer(c_int) :: iloc, jloc, iglob, jglob
  complex(C_FLOAT) :: tmp11
  logical :: loc_correct, elem_pass
  integer(c_int) :: nfound
  complex(C_COMPLEX), allocatable :: eig_vals(:)
  integer(c_int) :: i, error
  real(C_FLOAT), parameter :: refvals(16) = [ -1.01363347e+01_C_FLOAT, -8.85451267e+00_C_FLOAT, -4.73670736e+00_C_FLOAT, -4.41315701e+00_C_FLOAT, &
                                              -9.16672860e-01_C_FLOAT, -1.27597110e-01_C_FLOAT, -2.06397617e-02_C_FLOAT, -7.57303011e-04_C_FLOAT, &
                                              7.83462424e-04_C_FLOAT, 1.85921577e-02_C_FLOAT, 2.40734728e-01_C_FLOAT, 6.46603001e-01_C_FLOAT, &
                                              4.23501051e+00_C_FLOAT, 4.78593026e+00_C_FLOAT, 7.46607179e+00_C_FLOAT, 9.85701459e+00_C_FLOAT ]

  call MPI_Init(ierr)
  call setbuf(stdout, 0)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)

  Grows = 16_c_int
  Gcols = 16_c_int
  ProcX = 2_c_int
  ProcY = 2_c_int
  blockX = 8_c_int
  blockY = 8_c_int

  mpicxt = BLACScxtInit('R', MPI_COMM_WORLD, ProcX, ProcY)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  start = MPI_Wtime()
  Matrix = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt)
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  end = MPI_Wtime()

  if (my_rank == 0) then
    print *, "Init D mat : ", end - start
  end if

  if (.not. associated(Matrix)) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  error = 0

  ele_this_cpu = (Grows * Gcols) / size
  ele_rem = mod(Grows * Gcols, size)
  shift = ele_this_cpu * my_rank

  if (my_rank < ele_rem) then
    shift = shift + my_rank
    ele_this_cpu = ele_this_cpu + 1
  else
    shift = shift + ele_rem
  end if

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  start = MPI_Wtime()
  error = initiateSetQueue(Matrix, ele_this_cpu)
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  end = MPI_Wtime()
  if (my_rank == 0) then
    print *, "Init set : ", end - start
  end if

  if (error /= 0) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  start = MPI_Wtime()

  do i = 0, ele_this_cpu-1
    gele = i + shift
    ir = gele / Gcols
    jc = mod(gele, Gcols)
    tmp11 = csin(gele + cmplx(0.0_C_FLOAT, cos(gele)))

    gele = ir + jc * Gcols
    tmp11 = tmp11 + conjg(csin(gele + cmplx(0.0_C_FLOAT, cos(gele))))

    error = DMatSet(Matrix, ir, jc, tmp11)
    if (error /= 0) then
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
  end do

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  end = MPI_Wtime()
  if (my_rank == 0) then
    print *, "Setting mat : ", end - start
  end if

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  start = MPI_Wtime()

  error = ProcessSetQueue(Matrix)

  end = MPI_Wtime()
  if (my_rank == 0) then
    print *, "SetQueue Process : ", end - start
  end if

  if (error /= 0) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  loc_correct = .true.

  ! Check if the elements in the local block are correct
  do iloc = 0, dismat%ldims(1) - 1
    do jloc = 0, dismat%ldims(2) - 1
      iglob = INDXL2G(iloc, dismat%block_size(1), dismat%pids(1), 0, dismat%pgrid(1))
      jglob = INDXL2G(jloc, dismat%block_size(2), dismat%pids(2), 0, dismat%pgrid(2))

      gele = iglob * Gcols + jglob
      tmp11 = csin(gele + cmplx(0.0_C_FLOAT, cos(gele)))
      gele = jglob * Gcols + iglob
      tmp11 = tmp11 + conjg(csin(gele + cmplx(0.0_C_FLOAT, cos(gele))))

      if (abs(tmp11 - dismat%data(iloc * dismat%lda(1) + jloc * dismat%lda(2))) > 1.0e-6_C_FLOAT) then
        loc_correct = .false.
      end if
    end do
  end do

  call MPI_Reduce(loc_correct, elem_pass, 1, MPI_LOGICAL, MPI_LOR, 0, MPI_COMM_WORLD, ierr)

  if (my_rank == 0) then
    if (elem_pass) then
      print *, "Local elements set correctly. Passed"
    else
      print *, "Local elements NOT set correctly. Failed"
    end if
  end if

  Matrix_Z = init_D_Matrix(Grows, Gcols, blockX, blockY, mpicxt)

  if (.not. associated(Matrix_Z)) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  allocate(eig_vals(Grows))
  if (.not. allocated(eig_vals)) then
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  start = MPI_Wtime()
  if (my_rank == 0) then
    print *, "Diagonalization......."
  end if

#ifdef TEST_ELPA
  call Heev_Elpa(Matrix, eig_vals, Matrix_Z, -1, 2, null, 1)
#else
  call Geev(Matrix, eig_vals, null, Matrix_Z)
  nfound = 0
#endif

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  end = MPI_Wtime()
  if (my_rank == 0) then
    print *, "Diagonalization time : ", end - start
  end if

#ifdef PRINT_EIGS
  if (my_rank == 0) then
    do i = 0, Grows - 1
      print *, real(eig_vals(i)), aimag(eig_vals(i))
    end do
  end if
#else
  if (my_rank == 0) then
    logical :: pass
    pass = .true.
    do i = 0, Grows - 1
      if (abs(real(eig_vals(i)) - refvals(i)) > 1.0e-5_C_FLOAT .or. &
          abs(aimag(eig_vals(i))) > 1.0e-8_C_FLOAT) then
        pass = .false.
        exit
      end if
    end do

    if (pass) then
      print *, "Passed ;)"
    else
      print *, "Failed ;)"
    end if
  end if
#endif

  deallocate(eig_vals)
  call free_D_Matrix(Matrix_Z)
  call free_D_Matrix(Matrix)
  call BLACScxtFree(mpicxt)

  call MPI_Finalize(ierr)

end program main


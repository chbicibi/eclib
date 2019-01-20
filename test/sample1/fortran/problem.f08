module case0
use iso_c_binding
use util
use orbit_func
use orbit_base
use orbit_debri
implicit none

real(8) :: base_time
integer, allocatable :: ORDER(:)

contains

! ##############################################################################
! # IO
! ##############################################################################

subroutine initialize() bind(c, name="initialize")
    implicit none

    base_time = mjd(2018, 0d0)
    print *, 'initialize(fortran)', base_time
end subroutine initialize


subroutine init_debri(tle_file, rcs_file, n_debris) bind(c, name="init_debri")
    implicit none
    character(1, c_char), intent(in) :: tle_file(*)
    character(1, c_char), intent(in) :: rcs_file(*)
    integer(c_int), intent(out) :: n_debris
    integer :: i

    call load_debri(-1, fstring(tle_file), fstring(rcs_file))
    ! base_time = DEBRIS(1)%orbit%epc

    do i = 0, NUM_DEBRIS_ALL
      call DEBRIS(i)%orbit%move(base_time, 0)
    end do

    ! デブリデータ並べ替え
    call select_debri(NUM_DEBRIS_ALL, ORDER)
    ! do i = 0, 100
    !     print *, DEBRIS(order(i))%orbit%raan
    ! end do

    n_debris = NUM_DEBRIS_ALL
end subroutine init_debri


subroutine call_problem(iargs, n1, dargs, n2, result) bind(c, name="call_problem")
    ! ### 引数 ###
    ! iargs: デブリ番号の列 # [例] (1, 2, 3, 4) => 1 -> 2, 2 -> 3, 3 -> 4
    ! dargs(2n-1): 待機時間 [day]
    ! dargs(2n): 遷移時間 [s]
    ! result(1): ΔV [km/s]
    ! result(2): RCS [m^2]

    implicit none
    integer(c_int), intent(in), value :: n1, n2
    integer(c_int), intent(in) :: iargs(n1) ! 32bit整数型配列引数
    real(c_double), intent(in) :: dargs(n2) ! 64bit実数型配列引数
    real(c_double), intent(out) :: result(2)
    real(8) :: total_delta_v, total_rcs
    real(8) :: start, duration, dv
    integer :: i_dep, i_arr, i

    if (n2 /= 2*(n1-1)) then
        print *, "ArgumentError: invalid argment size"
        call exit(1)
    end if

    total_delta_v = 0d0
    start = base_time

    do i = 1, n1 - 1
        i_dep = ORDER(iargs(i))
        i_arr = ORDER(iargs(i+1))

      start = start + dargs(2*i-1)
      duration = dargs(2*i)

      call lambert(start, duration, DEBRIS(i_dep)%orbit, DEBRIS(i_arr)%orbit, dv)

      total_delta_v = total_delta_v + dv
    end do

    total_rcs = sum(DEBRIS(ORDER(iargs))%rcs)

    result(1) = total_delta_v
    result(2) = total_rcs
end subroutine call_problem

subroutine select_debri(n, order)
    implicit none
    integer, intent(in) :: n
    integer, intent(out), allocatable :: order(:)
    real(8), allocatable :: data(:)

    allocate(data(NUM_DEBRIS_ALL), source=DEBRIS(1:)%orbit%raan)
    allocate(order(0:n), source=[0, sort(data, n)])
end subroutine select_debri

! ##############################################################################
! # utilities
! ##############################################################################

function fstring(string)
    ! c言語の文字列ポインタをFortranの文字列配列に変換する
    implicit none
    character(1, c_char), intent(in) :: string(*)
    character(:, c_char), allocatable :: fstring
    integer :: len, i

    len = 1
    do while (string(len) /= c_null_char)
        len = len + 1
    end do
    len = len - 1
    allocate(character(len, c_char) :: fstring)
    do i = 1, len
      fstring(i:i) = string(i)
    end do
end function fstring

end module case0

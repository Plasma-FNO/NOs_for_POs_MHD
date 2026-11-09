! Repair only reversed intersections of two flux surfaces with the same polar curve.
! Keep both original intersections unless exact native roots restore their order.
subroutine opsplit_refine_reversed_pair(node_list,element_list,ielm,target,rc,zc,s,t,tau,rr,zz,ifail)
  use data_structure
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(type_node_list), intent(in) :: node_list
  type(type_element_list), intent(in) :: element_list
  integer, intent(in) :: ielm(2)
  real*8, intent(in) :: target(2),rc(4),zc(4)
  real*8, intent(inout) :: s(2),t(2),tau(2),rr(2),zz(2)
  integer, intent(out) :: ifail
  real*8 :: ss(2),tt(2),uu(2),rnew(2),znew(2)
  integer :: k,status
  ifail=0
  if (tau(2) >= tau(1)-1.d-12) return
  ifail=99
  if (target(1) == target(2)) return
  ss=s; tt=t; uu=tau
  do k=1,2
    call opsplit_refine_core_crossing(node_list,element_list,ielm(k),target(k), &
         rc,zc,ss(k),tt(k),uu(k),rnew(k),znew(k),status)
    if (status /= 0) return
  enddo
  if (.not. all(ieee_is_finite([ss,tt,uu,rnew,znew]))) return
  if (maxval(abs(uu)) > 1.d0+1.d-10 .or. uu(2) <= uu(1)+1.d-12) return
  s=ss; t=tt; tau=uu; rr=rnew; zz=znew
  ifail=0
end subroutine opsplit_refine_reversed_pair

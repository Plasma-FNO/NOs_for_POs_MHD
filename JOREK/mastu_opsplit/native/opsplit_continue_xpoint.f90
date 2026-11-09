! Continue a nearby stationary saddle; failure leaves the global search in charge.
subroutine opsplit_continue_xpoint(node_list,element_list,previous,R_axis,Z_axis,radius,side,ielm,s,t,found)
  use data_structure, only: type_node_list,type_element_list
  use mod_interp, only: interp,interp_RZ
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(type_node_list), intent(in) :: node_list
  type(type_element_list), intent(in) :: element_list
  real*8, intent(in) :: previous(2),R_axis,Z_axis,radius
  integer, intent(in) :: side
  integer, intent(out) :: ielm
  real*8, intent(out) :: s,t
  logical, intent(out) :: found
  integer :: seed,candidates(5),i,e,ifail
  real*8 :: ss,tt,seed_s,seed_t,R,Z,Rs,Rt,Zs,Zt,P,Ps,Pt,Pst,Pss,Ptt
  real*8 :: xerr,ferr,distance,best,det,jac
  found=.false.; ielm=0; s=0.d0; t=0.d0
  if (.not. all(ieee_is_finite(previous))) return
  call find_RZ(node_list,element_list,previous(1),previous(2),R,Z,seed,seed_s,seed_t,ifail)
  if (ifail /= 0 .or. seed < 1 .or. seed > element_list%n_elements) return
  candidates=[seed,element_list%element(seed)%neighbours]
  best=radius
  do i=1,5
    e=candidates(i)
    if (e < 1 .or. e > element_list%n_elements) cycle
    ss=.5d0; tt=.5d0
    if (i == 1) then
      ss=seed_s; tt=seed_t
    endif
    call mnewtax(node_list,element_list,e,ss,tt,xerr,ferr,ifail)
    if (ifail /= 0 .or. min(ss,tt) < 0.d0 .or. max(ss,tt) > 1.d0) cycle
    call interp(node_list,element_list,e,1,1,ss,tt,P,Ps,Pt,Pst,Pss,Ptt)
    call interp_RZ(node_list,element_list,e,ss,tt,R,Rs,Rt,Z,Zs,Zt)
    if (.not. all(ieee_is_finite([P,Ps,Pt,Pst,Pss,Ptt,R,Z,Rs,Rt,Zs,Zt]))) cycle
    if (abs(Ps)+abs(Pt) > 1.d-12) cycle
    det=Pss*Ptt-Pst*Pst
    if (det >= -64*epsilon(det)*(abs(Pss*Ptt)+Pst*Pst)) cycle
    jac=Rs*Zt-Rt*Zs
    if (abs(jac) <= 64*epsilon(jac)*sqrt((Rs*Rs+Zs*Zs)*(Rt*Rt+Zt*Zt))) cycle
    if (side*(Z-Z_axis) <= 0.d0) cycle
    if (sqrt((R-R_axis)**2+(Z-Z_axis)**2) <= radius) cycle
    distance=sqrt(sum(([R,Z]-previous)**2))
    if (distance >= best) cycle
    best=distance; ielm=e; s=ss; t=tt; found=.true.
  enddo
end subroutine opsplit_continue_xpoint

! Refine an approximate intersection using the original bicubic flux and geometry.
subroutine opsplit_refine_core_crossing(node_list,element_list,ielm,target,R_c,Z_c,s,t,tau,R_out,Z_out,ifail)
  use data_structure
  use mod_interp, only: interp, interp_RZ
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(type_node_list), intent(in) :: node_list
  type(type_element_list), intent(in) :: element_list
  integer, intent(in) :: ielm
  real*8, intent(in) :: target,R_c(4),Z_c(4)
  real*8, intent(inout) :: s,t,tau
  real*8, intent(out) :: R_out,Z_out
  integer, intent(out) :: ifail
  real*8 :: R,Rs,Rt,Z,Zs,Zt,P,Ps,Pt,Rp,dRp,Zp,dZp,det,denom
  real*8 :: residual(2),a(2),b(2),step(3),scale
  integer :: iteration
  ifail=99
  do iteration=1,16
    call interp_RZ(node_list,element_list,ielm,s,t,R,Rs,Rt,Z,Zs,Zt)
    call interp(node_list,element_list,ielm,1,1,s,t,P,Ps,Pt)
    call cub1D(R_c(1),R_c(2),R_c(3),R_c(4),tau,Rp,dRp)
    call cub1D(Z_c(1),Z_c(2),Z_c(3),Z_c(4),tau,Zp,dZp)
    residual=[R-Rp,Z-Zp]
    if (.not. all(ieee_is_finite([R,Z,P,Ps,Pt,residual]))) return
    if (maxval(abs(residual)) <= 1.d-11 .and. abs(P-target) <= 1.d-12) then
      if (min(s,t) < -1.d-10 .or. max(s,t) > 1.d0+1.d-10) return
      R_out=R; Z_out=Z
      ifail=0
      return
    endif
    det=Rs*Zt-Rt*Zs
    if (abs(det) <= 64*epsilon(det)*sqrt((Rs*Rs+Zs*Zs)*(Rt*Rt+Zt*Zt))) return
    a=[Zt*residual(1)-Rt*residual(2),Rs*residual(2)-Zs*residual(1)]/det
    b=[Zt*dRp-Rt*dZp,Rs*dZp-Zs*dRp]/det
    denom=Ps*b(1)+Pt*b(2)
    if (abs(denom) <= 64*epsilon(denom)*sqrt((Ps*Ps+Pt*Pt)*sum(b*b))) return
    step(3)=(target-P+Ps*a(1)+Pt*a(2))/denom
    step(1:2)=-a+b*step(3)
    if (.not. all(ieee_is_finite(step))) return
    scale=max(1.d0,maxval(abs(step))/.25d0)
    s=s+step(1)/scale; t=t+step(2)/scale; tau=tau+step(3)/scale
  enddo
end subroutine opsplit_refine_core_crossing

logical function opsplit_quad_is_concave(rr,zz)
  implicit none
  real*8,intent(in) :: rr(4),zz(4)
  real*8 :: dr(4),dz(4),cross(4),tol
  integer :: k,next
  do k=1,4
    next=mod(k,4)+1
    dr(k)=rr(next)-rr(k); dz(k)=zz(next)-zz(k)
  enddo
  do k=1,4
    next=mod(k,4)+1
    cross(k)=dr(k)*dz(next)-dz(k)*dr(next)
  enddo
  tol=64*epsilon(tol)*maxval(abs(cross))
  opsplit_quad_is_concave=minval(cross)<-tol .and. maxval(cross)>tol
end function opsplit_quad_is_concave

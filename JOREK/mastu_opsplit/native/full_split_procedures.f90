! The supported n=0 pilot has tauIC=TGNUM=hyperdiffusion=NEO=bootstrap=0.
! Channels multiply [v,v_R,v_Z] under R dR dZ; rows [psi,u,rho,p,Vpar].
subroutine opsplit_begin_step(step, rank_id)
  use, intrinsic :: ieee_arithmetic
  use phys_module, only: tstep,tstep_prev,time_evol_theta
  implicit none
  integer,intent(in) :: step,rank_id
  integer :: unit,ios,magic,ne,nq,stamp,ncoeff
  character(32) :: rank_string
  if (opsplit_dump_unit /= 0) close(opsplit_dump_unit)
  opsplit_dump_unit=0
  call get_environment_variable('OPSSPLIT_MODE',opsplit_mode)
  if (trim(opsplit_mode)=='') return
  call get_environment_variable('OPSSPLIT_SCHEME',opsplit_scheme)
  if(trim(opsplit_scheme)=='') opsplit_scheme='euler'
  if(trim(opsplit_scheme)/='euler' .and. trim(opsplit_scheme)/='cnab2' .and. trim(opsplit_scheme)/='sbdf2') error stop 'Unknown OpsSplit scheme'
  opsplit_history=trim(opsplit_scheme)/='euler' .and. opsplit_last_step/=0 &
       .and. (trim(opsplit_mode)=='numerical' .or. trim(opsplit_mode)=='neural')
  if(opsplit_last_step/=0 .and. step/=opsplit_last_step+1) error stop 'Nonconsecutive OpsSplit history'
  opsplit_last_step=step
  opsplit_weight=1.d0; opsplit_ratio=0.d0; opsplit_delta_weight=0.d0
  opsplit_mass_weight=1.d0; opsplit_history_weight=0.d0
  if(opsplit_history) then
    if(tstep_prev<=0.d0) error stop 'Invalid previous timestep'
    opsplit_weight=.5d0; opsplit_ratio=.5d0*tstep/tstep_prev; opsplit_delta_weight=.5d0
    if(trim(opsplit_scheme)=='sbdf2') then
      opsplit_weight=1.d0;opsplit_ratio=tstep/tstep_prev
      opsplit_mass_weight=(1.d0+2.d0*opsplit_ratio)/(1.d0+opsplit_ratio)
      ! Native delta_g/delta_s/delta_t already contain (h/h_previous)*Delta q.
      opsplit_delta_weight=1.d0
      opsplit_history_weight=opsplit_ratio/(1.d0+opsplit_ratio)
    endif
    opsplit_previous=opsplit_current
    opsplit_seen_previous=opsplit_seen
    opsplit_seen=.false.
  endif
  if (trim(opsplit_mode)/='export' .and. trim(opsplit_mode)/='numerical' .and. &
      trim(opsplit_mode)/='neural' .and. trim(opsplit_mode)/='recombine') error stop 'Unknown OpsSplit mode'
  if(trim(opsplit_mode)=='numerical' .or. trim(opsplit_mode)=='neural') then
    ! Boundary flux linearization must use the same implicit time weight.
    time_evol_theta=opsplit_weight
    if(rank_id==0) write(*,*) 'OpsSplit scheme/step/a0/theta:',trim(opsplit_scheme),step,opsplit_mass_weight,opsplit_weight
  endif
  if (.not.allocated(opsplit_vertices) .and. trim(opsplit_mode)/='export') then
    open(newunit=unit,file='opsplit_linear.bin',access='stream',form='unformatted',status='old',iostat=ios)
    if(ios/=0) error stop 'Missing frozen OpsSplit linear operator'
    read(unit) magic,ne,nq
    if((magic/=303046 .and. magic/=303361) .or. ne<1 .or. nq/=16) error stop 'Invalid OpsSplit linear header'
    ncoeff=46
    if(magic==303361) ncoeff=361
    allocate(opsplit_vertices(4,ne),opsplit_coeff(ncoeff,nq,ne),opsplit_prediction(3,5,nq,ne))
    allocate(opsplit_previous(3,5,nq,ne),opsplit_current(3,5,nq,ne), &
             opsplit_seen(nq,ne),opsplit_seen_previous(nq,ne))
    opsplit_current=0.d0; opsplit_previous=0.d0; opsplit_seen=.false.; opsplit_seen_previous=.false.
    read(unit) opsplit_vertices,opsplit_coeff
    close(unit)
    if(.not.all(ieee_is_finite(opsplit_coeff))) error stop 'Nonfinite frozen operator'
  endif
  if(trim(opsplit_mode)=='neural') then
    open(newunit=unit,file='opsplit_prediction.bin',access='stream',form='unformatted',status='old',iostat=ios)
    if(ios/=0) error stop 'Missing neural forcing'
    read(unit) magic,stamp,ne,nq
    if(magic/=303015 .or. stamp/=step .or. ne/=size(opsplit_vertices,2) .or. nq/=16) &
      error stop 'Stale or incompatible neural forcing'
    read(unit) opsplit_prediction
    close(unit)
    if(.not.all(ieee_is_finite(opsplit_prediction))) error stop 'Nonfinite neural forcing'
  endif
  if(trim(opsplit_mode)=='export') then
    write(rank_string,'(i0)') rank_id
    open(newunit=opsplit_dump_unit,file='opsplit_export_'//trim(rank_string)//'.bin', &
         access='stream',form='unformatted',status='new',iostat=ios)
    if(ios/=0) error stop 'OpsSplit export refuses overwrite; use one step'
  endif
end subroutine

integer function opsplit_element(vertices) result(found)
  implicit none
  integer,intent(in) :: vertices(4)
  integer :: e
  found=0
  if (.not.allocated(opsplit_vertices)) return
  do e=1,size(opsplit_vertices,2)
    if(all(opsplit_vertices(:,e)==vertices)) then
      found=e
      return
    endif
  enddo
  error stop 'Element absent from frozen OpsSplit geometry'
end function

pure subroutine opsplit_linear(q,qr,qz,R,a,c)
  implicit none
  real*8,intent(in) :: q(7),qr(7),qz(7),R,a(10)
  real*8,intent(out) :: c(3,5)
  c=0.d0
  c(1,1)=a(1)*q(3)/R**2
  c(2:3,2)=-a(2)*[qr(4),qz(4)]
  c(2,3)=-a(3)*qr(5)-a(4)*qz(5)
  c(3,3)=-a(4)*qr(5)-a(5)*qz(5)
  c(2,4)=-a(6)*qr(6)-a(7)*qz(6)
  c(3,4)=-a(7)*qr(6)-a(8)*qz(6)
  c(2,5)=-a(9)*qr(7)+a(10)*q(7)
  c(3,5)=-a(9)*qz(7)
end subroutine

pure subroutine opsplit_physical(q,qr,qz,R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar, &
                                jsource,Sp,Sh,muv,normalized,vsource,omegasource,c,a)
  implicit none
  real*8,intent(in) :: q(7),qr(7),qz(7),R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar
  real*8,intent(in) :: jsource,Sp,Sh,muv,vsource(2),omegasource(2)
  logical,intent(in) :: normalized
  real*8,intent(out) :: c(3,5),a(10)
  real*8 :: b(2),B2,gp(2),gu2,rhou,Tu,rhopsi,Tpsi,Vpsi,prho,ppsi,pj
  b=[qz(1),-qr(1)]/R
  B2=(F**2+qr(1)**2+qz(1)**2)/R**2
  gp=q(6)*[qr(5),qz(5)]+q(5)*[qr(6),qz(6)]
  gu2=qr(2)**2+qz(2)**2
  rhou=qr(5)*qz(2)-qz(5)*qr(2)
  Tu=qr(6)*qz(2)-qz(6)*qr(2)
  rhopsi=qr(5)*qz(1)-qz(5)*qr(1)
  Tpsi=qr(6)*qz(1)-qz(6)*qr(1)
  Vpsi=qr(7)*qz(1)-qz(7)*qr(1)
  prho=-rhopsi
  ppsi=gp(1)*qz(1)-gp(2)*qr(1)
  pj=qr(1)*qz(3)-qz(1)*qr(3)
  a(1:2)=[eta,mu]
  a(3:5)=[D+(Dpar-D)*b(1)**2/B2,(Dpar-D)*b(1)*b(2)/B2,D+(Dpar-D)*b(2)**2/B2]
  a(6:8)=[K+(Kpar-K)*b(1)**2/B2,(Kpar-K)*b(1)*b(2)/B2,K+(Kpar-K)*b(2)**2/B2]
  if(normalized) then
    a(9:10)=[muv,0.d0]
  else
    a(9:10)=[muv*F**2/R**2,2.d0*muv*F**2/R**3]
  endif
  call opsplit_linear(q,qr,qz,R,a,c)
  c(1,1)=eta*(q(3)-jsource)/R**2+(qr(1)*qz(2)-qz(1)*qr(2))/R
  c(1,2)=pj/R
  c(2,2)=c(2,2)-.5d0*R**3*gu2*qz(5)-R**3*q(5)*q(4)*qz(2)+R*gp(2)+R**2*Sp*qr(2)
  c(3,2)=c(3,2)+.5d0*R*gu2*(2.d0*R*q(5)+R**2*qr(5))+R**3*q(5)*q(4)*qr(2)-R*gp(1)+R**2*Sp*qz(2)
  c(1,3)=Sp+R*rhou+2.d0*q(5)*qz(2)-(q(7)*rhopsi+q(5)*Vpsi)/R
  c(1,4)=Sh+R*(q(5)*Tu+q(6)*rhou)+2.d0*gamma*q(5)*q(6)*qz(2) &
            -q(7)*(q(5)*Tpsi+q(6)*rhopsi)/R-gamma*q(5)*q(6)*Vpsi/R+(gamma-1.d0)*etah*(q(3)/R)**2
  c(1,5)=-ppsi/R-Sp*q(7)*B2-.5d0*q(7)**2*B2*prho/R
  c(2:3,5)=c(2:3,5)+.5d0*q(5)*q(7)**2*B2*b
  if(normalized) then
    c(2:3,5)=c(2:3,5)+muv*vsource
  else
    c(2:3,5)=c(2:3,5)+muv*2.d0*acos(-1.d0)*F*omegasource
  endif
end subroutine

pure subroutine opsplit_linear_complex(q,qr,qz,R,a,c)
  implicit none
  complex*16,intent(in) :: q(7),qr(7),qz(7),a(10)
  real*8,intent(in) :: R
  complex*16,intent(out) :: c(3,5)
  c=0.d0
  c(1,1)=a(1)*q(3)/R**2
  c(2:3,2)=-a(2)*[qr(4),qz(4)]
  c(2,3)=-a(3)*qr(5)-a(4)*qz(5)
  c(3,3)=-a(4)*qr(5)-a(5)*qz(5)
  c(2,4)=-a(6)*qr(6)-a(7)*qz(6)
  c(3,4)=-a(7)*qr(6)-a(8)*qz(6)
  c(2,5)=-a(9)*qr(7)+a(10)*q(7)
  c(3,5)=-a(9)*qz(7)
end subroutine

pure subroutine opsplit_physical_complex(q,qr,qz,R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar, &
                                jsource,Sp,Sh,muv,normalized,vsource,omegasource,c,a)
  implicit none
  complex*16,intent(in) :: q(7),qr(7),qz(7)
  real*8,intent(in) :: R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar
  real*8,intent(in) :: jsource,Sp,Sh,muv,vsource(2),omegasource(2)
  logical,intent(in) :: normalized
  complex*16,intent(out) :: c(3,5),a(10)
  complex*16 :: b(2),B2,gp(2),gu2,rhou,Tu,rhopsi,Tpsi,Vpsi,prho,ppsi,pj
  b=[qz(1),-qr(1)]/R
  B2=(F**2+qr(1)**2+qz(1)**2)/R**2
  gp=q(6)*[qr(5),qz(5)]+q(5)*[qr(6),qz(6)]
  gu2=qr(2)**2+qz(2)**2
  rhou=qr(5)*qz(2)-qz(5)*qr(2)
  Tu=qr(6)*qz(2)-qz(6)*qr(2)
  rhopsi=qr(5)*qz(1)-qz(5)*qr(1)
  Tpsi=qr(6)*qz(1)-qz(6)*qr(1)
  Vpsi=qr(7)*qz(1)-qz(7)*qr(1)
  prho=-rhopsi
  ppsi=gp(1)*qz(1)-gp(2)*qr(1)
  pj=qr(1)*qz(3)-qz(1)*qr(3)
  a(1:2)=[eta,mu]
  a(3:5)=[D+(Dpar-D)*b(1)**2/B2,(Dpar-D)*b(1)*b(2)/B2,D+(Dpar-D)*b(2)**2/B2]
  a(6:8)=[K+(Kpar-K)*b(1)**2/B2,(Kpar-K)*b(1)*b(2)/B2,K+(Kpar-K)*b(2)**2/B2]
  if(normalized) then
    a(9:10)=[muv,0.d0]
  else
    a(9:10)=[muv*F**2/R**2,2.d0*muv*F**2/R**3]
  endif
  call opsplit_linear_complex(q,qr,qz,R,a,c)
  c(1,1)=eta*(q(3)-jsource)/R**2+(qr(1)*qz(2)-qz(1)*qr(2))/R
  c(1,2)=pj/R
  c(2,2)=c(2,2)-.5d0*R**3*gu2*qz(5)-R**3*q(5)*q(4)*qz(2)+R*gp(2)+R**2*Sp*qr(2)
  c(3,2)=c(3,2)+.5d0*R*gu2*(2.d0*R*q(5)+R**2*qr(5))+R**3*q(5)*q(4)*qr(2)-R*gp(1)+R**2*Sp*qz(2)
  c(1,3)=Sp+R*rhou+2.d0*q(5)*qz(2)-(q(7)*rhopsi+q(5)*Vpsi)/R
  c(1,4)=Sh+R*(q(5)*Tu+q(6)*rhou)+2.d0*gamma*q(5)*q(6)*qz(2) &
            -q(7)*(q(5)*Tpsi+q(6)*rhopsi)/R-gamma*q(5)*q(6)*Vpsi/R+(gamma-1.d0)*etah*(q(3)/R)**2
  c(1,5)=-ppsi/R-Sp*q(7)*B2-.5d0*q(7)**2*B2*prho/R
  c(2:3,5)=c(2:3,5)+.5d0*q(5)*q(7)**2*B2*b
  if(normalized) then
    c(2:3,5)=c(2:3,5)+muv*vsource
  else
    c(2:3,5)=c(2:3,5)+muv*2.d0*acos(-1.d0)*F*omegasource
  endif
end subroutine


pure subroutine opsplit_tangent(q,qr,qz,R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar, &
                               jsource,Sp,Sh,muv,normalized,vsource,omegasource,jac)
  implicit none
  real*8,intent(in) :: q(7),qr(7),qz(7),R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar
  real*8,intent(in) :: jsource,Sp,Sh,muv,vsource(2),omegasource(2)
  logical,intent(in) :: normalized
  real*8,intent(out) :: jac(15,21)
  complex*16 :: jets(21),c(3,5),a(10)
  integer :: column
  do column=1,21
    jets=cmplx([q,qr,qz],0.d0,kind=8)
    jets(column)=jets(column)+cmplx(0.d0,1.d-30,kind=8)
    call opsplit_physical_complex(jets(1:7),jets(8:14),jets(15:21),R,F,gamma,eta,mu,etah,D,K,Dpar,Kpar, &
                                 jsource,Sp,Sh,muv,normalized,vsource,omegasource,c,a)
    jac(:,column)=reshape(aimag(c),[15])/1.d-30
  enddo
end subroutine

pure subroutine opsplit_frozen_action(q,qr,qz,R,frozen,c)
  implicit none
  real*8,intent(in) :: q(7),qr(7),qz(7),R,frozen(:)
  real*8,intent(out) :: c(3,5)
  if(size(frozen)==361) then
    c=reshape(matmul(reshape(frozen(47:361),[15,21]),[q,qr,qz]),[3,5])
  else
    call opsplit_linear(q,qr,qz,R,frozen(1:10),c)
  endif
end subroutine

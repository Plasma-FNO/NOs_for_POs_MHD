! Repair boundary-adjacent folds using only shared interior handles.
! Try the least contraction first; retain all node jets and physical boundary curves.
subroutine opsplit_limit_boundary_handle(nodes,elements)
  use data_structure, only: type_node_list,type_element_list
  use mod_parameters, only: n_order,n_degrees
  use mod_basisfunctions, only: basisfunctions
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(type_node_list), intent(in) :: nodes
  type(type_element_list), intent(inout) :: elements
  integer, allocatable :: counts(:),incident(:,:)
  integer :: e,v,n,j,k,m,other,neighbor,side,el,iv,iv2,np,attempt
  integer :: pair(2),vertices(2),dofs(2)
  real*8 :: saved(2,2),factor,H(4,n_degrees),hs(4,n_degrees,17,17),ht(4,n_degrees,17,17)
  logical :: good
  if (n_order /= 3) return
  allocate(counts(nodes%n_nodes),incident(4,nodes%n_nodes))
  counts=0;incident=0
  do e=1,elements%n_elements
    do v=1,4
      n=elements%element(e)%vertex(v)
      counts(n)=counts(n)+1
      if (counts(n)<=4) incident(counts(n),n)=e
    enddo
  enddo
  do j=1,17
    do k=1,17
      call basisfunctions(dble(j-1)/16,dble(k-1)/16,H,hs(:,:,j,k),ht(:,:,j,k))
    enddo
  enddo
  do e=1,elements%n_elements
    if (.not.any(counts(elements%element(e)%vertex(1:4))==2)) cycle
    if (positive_element(e)) cycle
    trials: do attempt=1,8
      factor=0.8d0**attempt
      do v=1,4
        n=elements%element(e)%vertex(v)
        if (n<=8) cycle
        if (counts(n)<2 .or. counts(n)>4) cycle
        do neighbor=1,2
          iv2=mod(v,4)+1
          if (neighbor==2) iv2=mod(v+2,4)+1
          other=elements%element(e)%vertex(iv2)
          if (other<=8) cycle
          np=0
          do m=1,counts(n)
            el=incident(m,n)
            do side=1,4
              iv=mod(side,4)+1
              if (.not. ((elements%element(el)%vertex(side)==n .and. elements%element(el)%vertex(iv)==other) .or. &
                         (elements%element(el)%vertex(iv)==n .and. elements%element(el)%vertex(side)==other))) cycle
              np=np+1
              if (np>2) exit
              pair(np)=el;vertices(np)=side
              if (elements%element(el)%vertex(iv)==n) vertices(np)=iv
              dofs(np)=2+mod(side+1,2)
            enddo
            if (np>2) exit
          enddo
          if (np/=2) cycle ! A physical boundary edge is never eligible.
          do m=1,2
            el=pair(m);iv=vertices(m)
            saved(m,:)=[elements%element(el)%size(iv,dofs(m)),elements%element(el)%size(iv,4)]
            elements%element(el)%size(iv,dofs(m))=saved(m,1)*factor
            elements%element(el)%size(iv,4)=elements%element(el)%size(iv,2)*elements%element(el)%size(iv,3)
          enddo
          good=.true.
          do m=1,counts(n)
            if (.not.positive_element(incident(m,n))) good=.false.
          enddo
          if (good) then
            write(*,*) 'Boundary adjacent interior handle limited:',n,pair,factor
            exit trials
          endif
          do m=1,2
            elements%element(pair(m))%size(vertices(m),dofs(m))=saved(m,1)
            elements%element(pair(m))%size(vertices(m),4)=saved(m,2)
          enddo
        enddo
      enddo
    enddo trials
  enddo
contains
  logical function positive_element(el) result(ok)
    integer, intent(in) :: el
    integer :: b,c,d,iv,nd
    real*8 :: rs(2),rt(2),jet(2),jac
    ok=.false.
    do b=1,17
      do c=1,17
        rs=0;rt=0
        do iv=1,4
          nd=elements%element(el)%vertex(iv)
          do d=1,4
            jet=nodes%node(nd)%x(1,d,:)*elements%element(el)%size(iv,d)
            rs=rs+hs(iv,d,b,c)*jet;rt=rt+ht(iv,d,b,c)*jet
          enddo
        enddo
        jac=rs(1)*rt(2)-rs(2)*rt(1)
        if (.not.ieee_is_finite(jac) .or. jac<=0.d0) return
      enddo
    enddo
    ok=.true.
  end function
end subroutine

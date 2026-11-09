! Limit only shared interior handles at a folded, three-cell boundary corner.
! Node jets and physical boundary curves remain unchanged. Failed trials roll back.
subroutine opsplit_limit_corner_handle(nodes,elements)
  use data_structure, only: type_node_list,type_element_list
  use mod_parameters, only: n_order,n_degrees
  use mod_basisfunctions, only: basisfunctions
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(type_node_list), intent(in) :: nodes
  type(type_element_list), intent(inout) :: elements
  integer, allocatable :: counts(:),incident(:,:)
  integer :: e,v,n,j,k,m,side,other,nb,ne,edges(6),multiplicity(6)
  integer :: pair(2),vertices(2),dofs(2),np,attempt
  real*8 :: saved(2,2),factor,H(4,n_degrees),hs(4,n_degrees,17,17),ht(4,n_degrees,17,17)
  logical :: good
  if (n_order /= 3) return
  allocate(counts(nodes%n_nodes),incident(3,nodes%n_nodes))
  counts=0;incident=0
  do e=1,elements%n_elements
    do v=1,4
      n=elements%element(e)%vertex(v)
      counts(n)=counts(n)+1
      if (counts(n)<=3) incident(counts(n),n)=e
    enddo
  enddo
  ! Include endpoints, unlike an interior quadrature-only fold screen.
  do j=1,17
    do k=1,17
      call basisfunctions(dble(j-1)/16,dble(k-1)/16,H,hs(:,:,j,k),ht(:,:,j,k))
    enddo
  enddo
  do n=9,nodes%n_nodes ! Reserved axis/X-point nodes are outside this repair.
    if (counts(n)/=3) cycle
    edges=0;multiplicity=0;ne=0
    do m=1,3
      e=incident(m,n)
      do side=1,4
        v=mod(side,4)+1
        if (elements%element(e)%vertex(side)==n) then
          other=elements%element(e)%vertex(v)
        else if (elements%element(e)%vertex(v)==n) then
          other=elements%element(e)%vertex(side)
        else
          cycle
        endif
        do j=1,ne
          if (edges(j)==other) exit
        enddo
        if (j>ne) then
          ne=ne+1;edges(ne)=other
        endif
        multiplicity(j)=multiplicity(j)+1
      enddo
    enddo
    if (count(multiplicity==1)/=2 .or. any(multiplicity>2)) cycle
    if (positive_corner(n)) cycle
    do nb=1,ne
      if (multiplicity(nb)/=2) cycle
      other=edges(nb)
      if (counts(other)/=4) cycle
      np=0
      do m=1,3
        e=incident(m,n)
        do side=1,4
          v=mod(side,4)+1
          if (.not. ((elements%element(e)%vertex(side)==n .and. elements%element(e)%vertex(v)==other) .or. &
                     (elements%element(e)%vertex(v)==n .and. elements%element(e)%vertex(side)==other))) cycle
          np=np+1;pair(np)=e;vertices(np)=side
          if (elements%element(e)%vertex(v)==n) vertices(np)=v
          dofs(np)=2+mod(side+1,2)
        enddo
      enddo
      if (np/=2) cycle
      do m=1,2
        saved(m,:)=[elements%element(pair(m))%size(vertices(m),dofs(m)), &
                    elements%element(pair(m))%size(vertices(m),4)]
      enddo
      good=.false.
      do attempt=1,8
        factor=0.8d0**attempt
        do m=1,2
          e=pair(m);v=vertices(m)
          elements%element(e)%size(v,dofs(m))=saved(m,1)*factor
          elements%element(e)%size(v,4)=elements%element(e)%size(v,2)*elements%element(e)%size(v,3)
        enddo
        good=positive_corner(n)
        if (good) exit
      enddo
      if (good) then
        write(*,*) 'Interior corner handle limited:',n,pair,factor
        exit
      endif
      do m=1,2
        elements%element(pair(m))%size(vertices(m),dofs(m))=saved(m,1)
        elements%element(pair(m))%size(vertices(m),4)=saved(m,2)
      enddo
    enddo
  enddo
contains
  logical function positive_corner(node) result(ok)
    integer, intent(in) :: node
    integer :: a,b,c,d,el,iv,nd
    real*8 :: rs(2),rt(2),jet(2),jac
    ok=.false.
    do a=1,3
      el=incident(a,node)
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
    enddo
    ok=.true.
  end function
end subroutine

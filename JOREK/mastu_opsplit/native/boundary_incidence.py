"""Prepare a parity-preserving replacement for repeated node-incidence scans."""
import re

def replace_boundary_incidence(source):
    end=source.index('end subroutine update_boundary_types')+len('end subroutine update_boundary_types')
    first,rest=source[:end],source[end:]
    anchor='  integer            :: i_times'
    if first.count(anchor)!=1:raise ValueError('unexpected boundary routine declarations')
    first=first.replace(anchor,anchor+'\n  integer, allocatable :: node_incidence(:)\n  integer :: incidence_node, incidence_vertex, incidence_element, max_node')
    anchor='  ! --- Initialise first!'
    if first.count(anchor)!=1:raise ValueError('unexpected boundary initializer')
    count='''  ! Count distinct incident elements once; duplicate vertices count only once.
  max_node=0
  do incidence_element=1,element_list%n_elements
    max_node=max(max_node,maxval(element_list%element(incidence_element)%vertex))
  enddo
  allocate(node_incidence(0:max_node))
  node_incidence=0
  do incidence_element=1,element_list%n_elements
    do incidence_vertex=1,4
      incidence_node=element_list%element(incidence_element)%vertex(incidence_vertex)
      if (any(element_list%element(incidence_element)%vertex(1:incidence_vertex-1) == incidence_node)) cycle
      node_incidence(incidence_node)=node_incidence(incidence_node)+1
    enddo
  enddo

'''
    first=first.replace(anchor,count+anchor)
    first,n=re.subn(r'call adjacent_elements\(element_list,node_list,([^\n]+)\)',r'call cached_adjacent_elements(\1)',first)
    if n!=10:raise ValueError(f'unexpected adjacent call count: {n}')
    anchor='end subroutine update_boundary_types'
    helper='''contains
  subroutine cached_adjacent_elements(ielm,iv,side_max,elm_sum)
    integer,intent(in) :: ielm,iv,side_max
    integer,intent(out) :: elm_sum
    elm_sum=min(side_max,node_incidence(element_list%element(ielm)%vertex(iv))-1)
  end subroutine cached_adjacent_elements
'''
    first=first.replace(anchor,helper+anchor)
    return first+rest

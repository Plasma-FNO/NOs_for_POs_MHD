if (opsplit_export_active) then
  write(opsplit_export_unit) opsplit_export_element_id, element%vertex, &
    ((nodes(opsplit_i)%index(opsplit_j),opsplit_j=1,4),opsplit_i=1,4), &
    (nodes(opsplit_i)%boundary,opsplit_i=1,4), &
    (merge(1,0,nodes(opsplit_i)%axis_node),opsplit_i=1,4), &
    (merge(1,0,nodes(opsplit_i)%constrained),opsplit_i=1,4)
  write(opsplit_export_unit) t_now, tstep, theta, zeta, GAMMA, HHZ(1,1)
  write(opsplit_export_unit) opsplit_points, opsplit_weights, opsplit_values, opsplit_grad, &
    opsplit_basis, opsplit_basis_grad, opsplit_selected, opsplit_weak, RHS, ELM
  flush(opsplit_export_unit)
  opsplit_export_seen(opsplit_export_element_id) = .true.
endif

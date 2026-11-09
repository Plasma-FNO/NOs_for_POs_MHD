if (opsplit_export_active) then
  if (xjac <= 0.d0) error stop 'OpsSplit native export folded geometry'
  opsplit_local = n_degrees*(i-1)+j
  opsplit_basis(opsplit_local,ms,mt) = v
  opsplit_basis_grad(:,opsplit_local,ms,mt) = [v_x,v_y]
  if (i == 1 .and. j == 1) then
    opsplit_points(:,ms,mt) = [BigR,y_g(ms,mt)]
    opsplit_weights(ms,mt) = xjac*wst
    opsplit_values(:,ms,mt) = [ps0,u0,zj0,w0,r0,T0,Vpar0]
    opsplit_grad(:,:,ms,mt) = reshape([ps0_x,ps0_y,u0_x,u0_y,zj0_x,zj0_y, &
      w0_x,w0_y,r0_x,r0_y,T0_x,T0_y,Vpar0_x,Vpar0_y],[2,7])
    opsplit_selected(1,ms,mt) = BigR*(r0_s*u0_t-r0_t*u0_s)/xjac + 2.d0*r0*u0_y &
      - Vpar0*(r0_s*ps0_t-r0_t*ps0_s)/(BigR*xjac) &
      - r0*(Vpar0_s*ps0_t-Vpar0_t*ps0_s)/(BigR*xjac)
    opsplit_selected(2,ms,mt) = BigR*(r0*(T0_s*u0_t-T0_t*u0_s) &
      + T0*(r0_s*u0_t-r0_t*u0_s))/xjac + 2.d0*GAMMA*r0*T0*u0_y &
      - Vpar0*(r0*(T0_s*ps0_t-T0_t*ps0_s)+T0*(r0_s*ps0_t-r0_t*ps0_s))/(BigR*xjac) &
      - GAMMA*r0*T0*(Vpar0_s*ps0_t-Vpar0_t*ps0_s)/(BigR*xjac)
  endif
  ! Independently retain the original source-coordinate selected weak terms.
  opsplit_weak(1,opsplit_local) = opsplit_weak(1,opsplit_local) + wst*( &
      v*BigR**2*(r0_s*u0_t-r0_t*u0_s) + v*2.d0*BigR*r0*u0_y*xjac &
      - v*Vpar0*(r0_s*ps0_t-r0_t*ps0_s) - v*r0*(Vpar0_s*ps0_t-Vpar0_t*ps0_s))
  opsplit_weak(2,opsplit_local) = opsplit_weak(2,opsplit_local) + wst*( &
      v*r0*BigR**2*(T0_s*u0_t-T0_t*u0_s) + v*T0*BigR**2*(r0_s*u0_t-r0_t*u0_s) &
      + v*r0*T0*2.d0*GAMMA*BigR*u0_y*xjac &
      - v*r0*Vpar0*(T0_s*ps0_t-T0_t*ps0_s) - v*T0*Vpar0*(r0_s*ps0_t-r0_t*ps0_s) &
      - v*r0*T0*GAMMA*(Vpar0_s*ps0_t-Vpar0_t*ps0_s))
endif

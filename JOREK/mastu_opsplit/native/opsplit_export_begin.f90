! One immutable pre-update frame per rank; no native coefficient is changed.
if (.not. opsplit_export_initialized) then
  call get_environment_variable('MASTU_NATIVE_EXPORT', opsplit_switch, status=opsplit_status)
  opsplit_export_enabled = opsplit_status == 0 .and. trim(opsplit_switch) == '1'
  opsplit_export_initialized = .true.
  if (opsplit_export_enabled) then
    if (n_tor /= 1 .or. n_plane /= 1 .or. n_degrees /= 4 .or. n_var /= 7) &
      error stop 'OpsSplit native export requires n=0 bicubic model303'
    call get_environment_variable('OMP_NUM_THREADS', opsplit_switch, status=opsplit_status)
    if (opsplit_status /= 0 .or. trim(opsplit_switch) /= '1') &
      error stop 'OpsSplit native export requires one OpenMP thread'
    allocate(opsplit_export_seen(n_elements_max))
    opsplit_export_seen = .false.
    call get_environment_variable('OMPI_COMM_WORLD_RANK', opsplit_rank_text, status=opsplit_status)
    if (opsplit_status /= 0) error stop 'OpsSplit native export needs explicit MPI rank'
    read(opsplit_rank_text,*) opsplit_rank
    write(opsplit_filename,'(A,I6.6,A)') 'opsplit-native-rank-',opsplit_rank,'.bin'
    open(newunit=opsplit_export_unit,file=trim(opsplit_filename),access='stream', &
         form='unformatted',status='new',action='write')
    write(opsplit_export_unit) 3031601, n_gauss, n_var, 16, 4, 8, 1, opsplit_rank
  endif
endif
opsplit_export_active = .false.
if (opsplit_export_enabled .and. index_now == 1) then
  if (opsplit_export_element_id < 1 .or. opsplit_export_element_id > n_elements_max) &
    error stop 'OpsSplit native export invalid element identity'
  opsplit_export_active = .not. opsplit_export_seen(opsplit_export_element_id)
  if (opsplit_export_active) then
    if (tid /= 1 .or. use_fft .or. i_tor_min /= 1 .or. i_tor_max /= 1) &
      error stop 'OpsSplit native export unsupported assembly branch'
    ! RHS/dt is a spatial quantity only when history and dt-dependent TG vanish.
    if (zeta /= 0.d0 .or. any(TGNUM /= 0.d0) .or. tstep <= 0.d0) &
      error stop 'OpsSplit native export needs zero history and zero Taylor-Galerkin'
    if (any(HHZ_p /= 0.d0) .or. HHZ(1,1) /= 1.d0) &
      error stop 'OpsSplit native export toroidal normalization is not supported'
    if (any([(nodes(opsplit_i)%constrained,opsplit_i=1,n_vertex_max)])) &
      error stop 'OpsSplit native export needs native constraint elimination'
    opsplit_weak = 0.d0
  endif
endif

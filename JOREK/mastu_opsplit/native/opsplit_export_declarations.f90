logical :: opsplit_export_active
integer :: opsplit_local, opsplit_rank, opsplit_status, opsplit_i, opsplit_j
character(len=64) :: opsplit_switch, opsplit_rank_text, opsplit_filename
real*8 :: opsplit_points(2,n_gauss,n_gauss), opsplit_weights(n_gauss,n_gauss)
real*8 :: opsplit_values(7,n_gauss,n_gauss), opsplit_grad(2,7,n_gauss,n_gauss)
real*8 :: opsplit_basis(16,n_gauss,n_gauss), opsplit_basis_grad(2,16,n_gauss,n_gauss)
real*8 :: opsplit_selected(2,n_gauss,n_gauss), opsplit_weak(2,16)

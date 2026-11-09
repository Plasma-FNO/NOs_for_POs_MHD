"""Prepare an isolated, source-pinned native n=0 full split; never edit installation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/single_writer_build_20261102/source')
HASH = '16928a171df02b23dc2ea2371897f4540f2966d18bbcc85882d61d0bc9b7f732'
ROOT = Path(__file__).resolve().parents[1]


def patch(text):
    if hashlib.sha256(text.encode()).hexdigest() != HASH:
        raise ValueError('Unrecognized native source')
    declarations = '''
  character(32), save :: opsplit_mode='',opsplit_scheme='euler'
  integer,save :: opsplit_last_step=0
  logical,save :: opsplit_history=.false.
  real*8,save :: opsplit_weight=1.d0,opsplit_ratio=0.d0,opsplit_delta_weight=0.d0,opsplit_mass_weight=1.d0,opsplit_history_weight=0.d0
  real*8,allocatable,save :: opsplit_previous(:,:,:,:),opsplit_current(:,:,:,:)
  logical,allocatable,save :: opsplit_seen(:,:),opsplit_seen_previous(:,:)
  integer, save :: opsplit_dump_unit=0
  integer, allocatable, save :: opsplit_vertices(:,:)
  real*8, allocatable, save :: opsplit_coeff(:,:,:),opsplit_prediction(:,:,:,:)
'''
    text = text.replace('  implicit none\n\ncontains', '  implicit none\n'+declarations+'\ncontains', 1)
    text = text.replace('contains\n', 'contains\n'+(ROOT/'native/full_split_procedures.f90').read_text()+'\n', 1)
    text = text.replace('use mod_sources\n','use mod_sources\nuse, intrinsic :: ieee_arithmetic, only: ieee_is_finite\n',1)
    text = text.replace('integer    :: i, j, ms,', 'integer :: os_e,os_q,os_row\nreal*8 :: os_tangent(15,21),os_c(3,5),os_l(3,5),os_a(10),os_mid(7),os_mid_grad(2),os_mid_rho,os_n(3,5),os_qv(7),os_qr(7),os_qz(7),os_check(5),os_rhs(5),os_roundoff(5),os_mass_action(5),os_jac(3,5),os_delta(7),os_dr(7),os_dz(7)\ninteger :: os_rows(5)=[1,2,5,6,7]\ninteger    :: i, j, ms,', 1)
    text = text.replace('theta = time_evol_theta', '''theta = time_evol_theta
if(trim(opsplit_mode)/='') then
  if(n_tor/=1 .or. n_plane/=1 .or. n_gauss/=4 .or. tauIC/=0.d0 .or. NEO .or. bootstrap .or. use_pellet &
     .or. any(TGNUM/=0.d0) .or. eta_num/=0.d0 .or. visco_num/=0.d0 .or. D_perp_num/=0.d0 &
     .or. ZK_perp_num/=0.d0 .or. visco_par_num/=0.d0 .or. time_evol_zeta/=0.d0) &
     error stop 'Unsupported physics or history in full OpsSplit pilot'
  os_e=opsplit_element(element%vertex)
endif
if(trim(opsplit_mode)=='numerical' .or. trim(opsplit_mode)=='neural') theta=0.d0''', 1)
    anchor='            !###################################################################################################\n            !#  RHS equations end'
    insert='''            if(trim(opsplit_mode)/='') then
              os_q=(ms-1)*n_gauss+mt
              os_qv=[ps0,u0,zj0,w0,r0,T0,Vpar0]
              if(.not.all(ieee_is_finite(os_qv)) .or. r0<=0.d0 .or. T0<=0.d0) &
                 error stop 'Invalid current native quadrature state in OpsSplit'
              os_qr=[ps0_x,u0_x,zj0_x,w0_x,r0_x,T0_x,Vpar0_x]
              os_qz=[ps0_y,u0_y,zj0_y,w0_y,r0_y,T0_y,Vpar0_y]
              if(.not.all(ieee_is_finite([os_qr,os_qz]))) error stop 'Nonfinite native gradients'
              if(trim(opsplit_mode)/='neural') then
                call opsplit_physical(os_qv,os_qr,os_qz,BigR,F0,gamma,eta_T,visco_T,eta_T_ohm, &
                     D_prof,ZK_prof,D_par_local,ZKpar_T,current_source(ms,mt),particle_source(ms,mt),heat_source(ms,mt), &
                     visco_par,normalized_velocity_profile,[Vt0_x,Vt0_y],[Omega_tor0_x,Omega_tor0_y],os_c,os_a)
                os_check=matmul([v,v_x,v_y],os_c)*BigR*xjac*tstep
                os_rhs=[rhs_ij_1,rhs_ij_2,rhs_ij_5,rhs_ij_6,rhs_ij_7]
                ! Bound subtraction roundoff using the induction operands before cancellation.
                os_roundoff=0.d0
                os_roundoff(1)=512.d0*epsilon(1.d0)*abs(v*xjac*tstep)*( &
                   abs(eta_T)*(abs(zj0)+abs(current_source(ms,mt)))/BigR &
                   +abs(ps0_x*u0_y)+abs(ps0_y*u0_x))
                ! Cancellation-aware weak-channel scale; trajectory parity uses a separate fixed gate.
                if(any(abs(os_check-os_rhs)>1.d-9*(abs(os_rhs)+ &
                   matmul(abs([v,v_x,v_y]),abs(os_c))*abs(BigR*xjac*tstep)+1.d-25)+os_roundoff)) then
                  write(*,*) 'OpsSplit identity coordinates:',BigR,y_g(ms,mt),i,j,ms,mt
                  write(*,*) 'original:',os_rhs
                  write(*,*) 'reconstructed:',os_check
                  error stop 'Full weak coefficient identity failed'
                endif
              endif
              if(trim(opsplit_mode)=='export' .and. i==1 .and. j==1 .and. im==1) then
                call opsplit_tangent(os_qv,os_qr,os_qz,BigR,F0,gamma,eta_T,visco_T,eta_T_ohm, &
                     D_prof,ZK_prof,D_par_local,ZKpar_T,current_source(ms,mt),particle_source(ms,mt),heat_source(ms,mt), &
                     visco_par,normalized_velocity_profile,[Vt0_x,Vt0_y],[Omega_tor0_x,Omega_tor0_y],os_tangent)
                write(opsplit_dump_unit) element%vertex,os_q, &
                   [BigR,y_g(ms,mt),xjac,os_qv,os_qr,os_qz,os_a,reshape(os_c,[15]),reshape(os_tangent,[315])]
              endif
              if(trim(opsplit_mode)/='export') then
                ! Centering avoids subtracting the large equilibrium current/source balance.
                call opsplit_frozen_action(os_qv-opsplit_coeff(11:17,os_q,os_e), &
                     os_qr-opsplit_coeff(18:24,os_q,os_e),os_qz-opsplit_coeff(25:31,os_q,os_e), &
                     BigR,opsplit_coeff(:,os_q,os_e),os_l)
                os_l=os_l+reshape(opsplit_coeff(32:46,os_q,os_e),[3,5])
                if(trim(opsplit_mode)=='neural') then
                  os_n=opsplit_prediction(:,:,os_q,os_e)
                else
                  ! Numerical oracle follows the identical L + N replacement path.
                  os_n=os_c-os_l
                endif
                if(i==1 .and. j==1 .and. im==1) then
                  opsplit_current(:,:,os_q,os_e)=os_n
                  opsplit_seen(os_q,os_e)=.true.
                endif
                os_c=os_l+os_n
                if(opsplit_history) then
                  if(.not.opsplit_seen_previous(os_q,os_e)) error stop 'Missing nonlinear history'
                  os_c=os_c+opsplit_ratio*(os_n-opsplit_previous(:,:,os_q,os_e))
                endif
                os_mid=os_qv+opsplit_delta_weight*delta_g(mp,1:7,ms,mt)
                os_mid_grad=[ps0_x,ps0_y]+opsplit_delta_weight*[delta_ps_x,delta_ps_y]
                os_mid_rho=corr_neg_dens1(os_mid(5))
                os_rhs=matmul([v,v_x,v_y],os_c)*BigR*xjac*tstep
                if(opsplit_history_weight/=0.d0) then
                  os_mass_action(1)=v*delta_g(mp,1,ms,mt)/BigR**2
                  os_mass_action(2)=-BigR**2*os_mid_rho*(v_x*delta_u_x+v_y*delta_u_y)
                  os_mass_action(3)=v*delta_g(mp,5,ms,mt)
                  os_mass_action(4)=v*(os_mid(6)*delta_g(mp,5,ms,mt)+os_mid_rho*delta_g(mp,6,ms,mt))
                  os_mass_action(5)=v*(os_mid(5)*os_mid(7)*dot_product(os_mid_grad,[delta_ps_x,delta_ps_y]) &
                       +os_mid_rho*F0**2*delta_g(mp,7,ms,mt))/BigR**2
                  os_rhs=os_rhs+opsplit_history_weight*BigR*xjac*os_mass_action
                endif
                rhs_ij_1=os_rhs(1); rhs_ij_2=os_rhs(2); rhs_ij_5=os_rhs(3)
                rhs_ij_6=os_rhs(4); rhs_ij_7=os_rhs(5)
              endif
            endif

'''
    if text.count(anchor)!=1: raise ValueError('RHS anchor changed')
    text=text.replace(anchor, insert+anchor)
    # Native mass and auxiliary constraints remain; physical Jacobian theta is zero.
    # Add exactly -dt L to that coupled matrix (first-order IMEX Euler).
    anchor='                  ! --- Fill up the matrix'
    if text.count(anchor)!=1:
        # Pinned source uses a different comment: find first FFT branch after all amat expressions.
        begin=text.index('                  amat_77_kn')
        pos=text.index('                  if (use_fft) then',begin)
    else: pos=text.index(anchor)
    insert='''                  if(trim(opsplit_mode)=='numerical' .or. trim(opsplit_mode)=='neural') then
'''
    insert+='''                    if(opsplit_history) then
                      amat_11=opsplit_mass_weight*v*psi/BigR*xjac
                      amat_55=opsplit_mass_weight*v*rho*BigR*xjac
                      amat_22=-opsplit_mass_weight*BigR**3*os_mid_rho*(v_x*u_x+v_y*u_y)*xjac
                      amat_65=opsplit_mass_weight*v*rho*os_mid(6)*BigR*xjac
                      amat_66=opsplit_mass_weight*v*os_mid_rho*T*BigR*xjac
                      amat_71=opsplit_mass_weight*v*os_mid(5)*os_mid(7)/BigR*dot_product(os_mid_grad,[psi_x,psi_y])*xjac
                      amat_77=opsplit_mass_weight*v*Vpar*os_mid_rho*F0**2/BigR*xjac
                    endif
'''
    for col in range(1,8):
        insert+=f"""                    if(size(opsplit_coeff,1)==361) then
                      os_jac=reshape(opsplit_coeff({47+15*(col-1)}:{46+15*col},os_q,os_e),[3,5])*psi &
                            +reshape(opsplit_coeff({47+15*(col+6)}:{46+15*(col+7)},os_q,os_e),[3,5])*psi_x &
                            +reshape(opsplit_coeff({47+15*(col+13)}:{46+15*(col+14)},os_q,os_e),[3,5])*psi_y
                    else
                      os_delta=0.d0; os_dr=0.d0; os_dz=0.d0
                      os_delta({col})=psi; os_dr({col})=psi_x; os_dz({col})=psi_y
                      call opsplit_linear(os_delta,os_dr,os_dz,BigR,opsplit_coeff(1:10,os_q,os_e),os_jac)
                    endif
"""
        for row,slot in [(1,1),(2,2),(5,3),(6,4),(7,5)]:
            insert+=f"                    amat_{row}{col}=amat_{row}{col}-dot_product([v,v_x,v_y],os_jac(:,{slot}))*BigR*xjac*tstep*opsplit_weight\n"
    insert+='                  endif\n\n'
    text=text[:pos]+insert+text[pos:]
    # The original nonlinear Jacobian is identically disabled in these modes.
    # Assemble only its native temporal/constraint part, avoiding unused work.
    begin=text.index('                  amat_11 = ')
    end=text.index('                  if (use_fft) then',begin)
    mass="                  if(trim(opsplit_mode)=='numerical' .or. trim(opsplit_mode)=='neural') then\n"
    for row in range(1,8):
        mass+='                    '+'; '.join(f'amat_{row}{col}=0.d0' for col in range(1,8))+'\n'
    mass+="""                    amat_11=v*psi/BigR*xjac
                    amat_22=-BigR**3*r0_corr*(v_x*u_x+v_y*u_y)*xjac
                    amat_31=(v_x*psi_x+v_y*psi_y)/BigR*xjac
                    amat_33=v*zj/BigR*xjac
                    amat_42=(v_x*u_x+v_y*u_y)*BigR*xjac
                    amat_44=v*w*BigR*xjac
                    amat_55=v*rho*BigR*xjac
                    amat_65=v*rho*T0*BigR*xjac
                    amat_66=v*r0_corr*T*BigR*xjac
                    amat_71=v*r0*vpar0/BigR*(ps0_x*psi_x+ps0_y*psi_y)*xjac
                    amat_77=v*Vpar*r0_corr*F0**2/BigR*xjac
                  else
"""
    text=text[:begin]+mass+text[begin:end]+'                  endif\n\n'+text[end:]
    return text


def prepare(output):
    output=Path(output); output.mkdir(exist_ok=False,parents=True)
    source=output/'source'
    shutil.copytree(SOURCE,source,symlinks=True)
    path=source/'models/model303/mod_elt_matrix_fft.f90'
    original=path.read_text(); modified=patch(original);path.write_text(modified)
    main=source/'jorek2_main.f90'; main_original=main.read_text(); t=main_original
    t=t.replace('  use construct_matrix_mod,', '  use mod_elt_matrix_fft, only: opsplit_begin_step\n  use construct_matrix_mod,',1)
    t=t.replace('  integer                  :: my_id','  character(32) :: os_mode\n  integer :: os_status\n  integer                  :: my_id',1)
    anchor='    call construct_matrix(mhd_sim, mhd_sim%local_elms, mhd_sim%n_local_elms, a_mat, rhs_vec, harmonic_matrix=.false.)'
    assert t.count(anchor)==1
    t=t.replace(anchor,'''    call get_environment_variable('OPSSPLIT_MODE',os_mode)
    if(trim(os_mode)=='neural' .and. my_id==0) then
      call export_restart(mhd_sim%node_list,mhd_sim%element_list,'opsplit_state')
      call execute_command_line('./opsplit_infer.sh',exitstat=os_status)
      if(os_status/=0) error stop 'OpsSplit neural inference failed'
    endif
    if(trim(os_mode)=='neural') call MPI_Barrier(MPI_COMM_WORLD,ierr)
    call opsplit_begin_step(index_now,my_id)
'''+anchor)
    main.write_text(t)
    import difflib
    (output/'full_split.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/models/model303/mod_elt_matrix_fft.f90',tofile='b/models/model303/mod_elt_matrix_fft.f90'))+''.join(difflib.unified_diff(main_original.splitlines(True),t.splitlines(True),fromfile='a/jorek2_main.f90',tofile='b/jorek2_main.f90')))
    (output/'protocol.json').write_text(json.dumps(dict(original_sha256=HASH,patched_sha256=hashlib.sha256(modified.encode()).hexdigest(),supported_schemes=['euler','cnab2','sbdf2'],scheme='native temporal/constraint solve, frozen transport L implicit, complete centered nonlinear volume forcing explicit',boundary='native coupled boundary operators retained',physics='axisymmetric model303 pilot only; unsupported terms fail closed',training_ready=False),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    prepare(parser.parse_args().output)

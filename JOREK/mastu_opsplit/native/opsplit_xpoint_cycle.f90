! Trigger continuation only for a resolved, stagnant two-cycle in saddle positions.
logical function opsplit_xpoint_cycle(points,updates,length_scale,accuracy) result(cycling)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  real*8, intent(in) :: points(2,2,4),updates(4),length_scale,accuracy
  real*8 :: separation,repeat_error
  cycling=.false.
  if (.not. all(ieee_is_finite(points)) .or. .not. all(ieee_is_finite(updates))) return
  if (.not. ieee_is_finite(length_scale) .or. .not. ieee_is_finite(accuracy)) return
  if (length_scale <= 0.d0 .or. accuracy <= 0.d0 .or. any(updates <= accuracy)) return
  if (updates(3) < .9d0*updates(1) .or. updates(4) < .9d0*updates(2)) return
  separation=min(sqrt(sum((points(:,:,2)-points(:,:,1))**2)), &
                 sqrt(sum((points(:,:,3)-points(:,:,2))**2)), &
                 sqrt(sum((points(:,:,4)-points(:,:,3))**2)))
  if (separation <= 1.d-8*length_scale .or. separation >= .015d0*length_scale) return
  repeat_error=max(sqrt(sum((points(:,:,3)-points(:,:,1))**2)), &
                   sqrt(sum((points(:,:,4)-points(:,:,2))**2)))
  cycling=repeat_error < .1d0*separation
end function opsplit_xpoint_cycle

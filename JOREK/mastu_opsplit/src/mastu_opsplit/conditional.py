"""Initial-field L-SplineFNO with PDE-parameter and elapsed-time FiLM conditioning."""
import h5py
import numpy as np
import torch
from torch import nn

from .learning import FieldSplineOperator


class TimeConditionalSplineModel(nn.Module):
    """Query each time from q0; no autoregressive state or time integrator.

    The residual form q0 + (tau/T) * correction enforces the initial condition.
    Native mass projection and physical guards still perform spatial solves.
    """
    family = 'conditional'
    direct_query = True

    def __init__(self, space, model_config, time_scale_seconds, *, physical_guard):
        super().__init__()
        if not np.isfinite(time_scale_seconds) or time_scale_seconds <= 0:
            raise ValueError('Time scale must be finite and positive')
        self.space, self.physical_guard = space, physical_guard
        self.time_scale_seconds = float(time_scale_seconds)
        self.operator = FieldSplineOperator(7, 7, space.geometry.bounds, model_config,
                                           condition_transport=True, condition_time=True)
        self.register_buffer('coordinates', torch.tensor(
            space.geometry.quadrature_rz.reshape(-1, 2), dtype=torch.float64))

    def forward(self, initial, parameters, *, elapsed_seconds):
        if initial.ndim != 3 or initial.shape[1:] != (len(self.space.geometry.dof_ids), 7):
            raise ValueError('Initial state must have shape [batch,native_dof,7]')
        if parameters.shape != (initial.shape[0], self.operator.conditioning_parameters):
            raise ValueError('PDE parameters must match the batch and configured width')
        elapsed = torch.as_tensor(elapsed_seconds, device=initial.device, dtype=initial.dtype)
        if elapsed.ndim == 0:
            elapsed = elapsed.expand(initial.shape[0])
        if elapsed.shape != (initial.shape[0],) or not torch.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError('Elapsed seconds must be finite, nonnegative, scalar or [batch]')
        tau = elapsed / self.time_scale_seconds
        condition = torch.cat((parameters.to(initial), tau[:, None]), dim=1)
        values = self.space.sample(initial)
        correction = self.operator(values.float(), self.coordinates, condition.float()).to(initial.dtype)
        proposal = initial + tau[:, None, None] * self.space.project(correction)
        self.last_raw_prediction = proposal.detach()
        guarded = self.physical_guard(initial, proposal)
        # Preserve the supplied initial coefficients exactly, including roundoff in closure.
        return torch.where(tau[:, None, None] == 0, initial, guarded)


def initial_time_sample(data, case, query_frames, *, stride, start_frame=0):
    """Read only q0 and requested frames, using the existing training normalization."""
    frames = np.asarray(query_frames)
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu' or \
            (frames < 1).any() or type(stride) is not int or stride < 1:
        raise ValueError('Query frames and stride must be positive integers')
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError('Start frame must be a nonnegative integer')
    frames = start_frame + np.concatenate(([0], frames * stride))
    if frames.max() >= len(data.time_seconds):
        raise ValueError('Query time exceeds the available trajectory')
    unique, inverse = np.unique(frames, return_inverse=True)
    with h5py.File(case['resolved_path']) as stream:
        states = stream['native_state'][unique][inverse]
    if not np.isfinite(states).all():
        raise ValueError('Nonfinite initial/time-query state')
    return dict(state=torch.from_numpy((states-data.mean)/data.scale),
                parameters=torch.ones(1), case_id=case['case_id'], start_frame=start_frame,
                time_seconds=torch.from_numpy(data.time_seconds[frames].copy()))

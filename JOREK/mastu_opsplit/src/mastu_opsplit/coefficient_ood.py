"""Held-out density-diffusivity data and an explicitly coefficient-aware split.

For constant D_perp, model303 contains D_perp * cylindrical_laplacian(rho).
The magnetic anisotropy correction and other omitted physics remain in the
coefficient-conditioned neural residual. Spline differentiation is analytic;
learned scatter still approximates the physical field. This is not the complete
model303 linearization and does not make old continuity-only checkpoints valid.
"""
from dataclasses import fields
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .euler import EulerSplineModel
from .euler_dataset import EulerDataset, EulerWindows
from .numerical import NativeGeometry


class DensityDiffusionEulerModel(EulerSplineModel):
    """Continuity OpsSplit plus a known D_perp term; AR/NODE share its scalar input."""
    def __init__(self, family, space, model_config, *, mean, scale, code_interval,
                 baseline_density_diffusivity=1e-5, physical_linear=None, physical_guard=None):
        super().__init__(family, space, model_config, linear_branch='lspline_continuity',
                         physical_linear=physical_linear, physical_guard=physical_guard)
        if not np.isfinite(code_interval) or code_interval <= 0 or \
           not np.isfinite(baseline_density_diffusivity) or baseline_density_diffusivity <= 0:
            raise ValueError('Require positive physical diffusivity and native code-time interval')
        mean, scale = np.asarray(mean), np.asarray(scale)
        if mean.shape != (len(space.geometry.dof_ids),7) or scale.shape != (7,) or \
           not np.isfinite(mean).all() or not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError('Invalid training normalization')
        # Include the training mean: diffusion acts on physical rho, not just its anomaly.
        offset = np.einsum('eql,elc->eqc', space.geometry.basis,
                           (mean/scale)[space.geometry.dof_map]).reshape(-1,7)
        self.register_buffer('physical_mean_over_scale', torch.tensor(offset, dtype=torch.float64))
        self.register_buffer('density_diffusion_per_frame', torch.tensor(
            code_interval*baseline_density_diffusivity, dtype=torch.float64))

    def density_rhs(self, state, parameters):
        if self.family != 'opsplit':
            raise ValueError('Explicit density diffusion belongs to OpsSplit')
        if parameters.shape != (state.shape[0],1) or not torch.all(torch.isfinite(parameters) & (parameters > 0)):
            raise ValueError('Expected positive density diffusivity multiplier [B,1]')
        values = self.space.sample(state) + self.physical_mean_over_scale
        source = self.operator.normalized_coordinates(self.coordinates, state.shape[0]).to(values.float())
        coefficients = self.operator.backbone._encode(values.float().transpose(1,2), source)[:,4:5]
        laplace = (self.derivatives.laplace(coefficients)
                   + self.derivatives(coefficients,(1,0))/self.coordinates[None,:,0:1].to(coefficients))
        result = torch.zeros_like(values)
        result[...,4:5] = (laplace.to(state.dtype)*parameters[:,None,:].to(state.dtype)
                          * self.density_diffusion_per_frame)
        return self.space.project(result)

    def rhs(self, state, parameters):
        result = super().rhs(state, parameters)
        if self.family == 'opsplit':
            result = result + self.density_rhs(state, parameters)
        return result


class GammaEulerModel(DensityDiffusionEulerModel):
    """Explicit gamma in temperature compression; shared [D/D0, gamma] conditioning.

    Parent ID windows carry only D/D0; their gamma is the known baseline 5/3.
    OOD windows carry both entries. The nonlinear residual also sees gamma,
    including its omitted pressure, heating and sheath effects.
    """
    def __init__(self, family, space, model_config, **kwargs):
        super().__init__(family, space, dict(model_config, conditioning_parameters=2), **kwargs)

    def rhs(self, state, parameters):
        if parameters.shape == (state.shape[0], 1):
            parameters = torch.cat((parameters, torch.full_like(parameters, 5/3)), dim=1)
        if parameters.shape != (state.shape[0], 2) or not torch.all(torch.isfinite(parameters)) or \
           not torch.all((parameters[:,0] > 0) & (parameters[:,1] > 0) & (parameters[:,1] <= 2)):
            raise ValueError('Expected [D/D0, gamma] with D/D0 > 0 and 0 < gamma <= 2')
        values = self.space.sample(state)
        neural = self.operator(values.float(), self.coordinates, parameters.float()).to(state.dtype)
        result = self.space.project(neural)
        if self.family == 'opsplit':
            result = result + self.linear_rhs(state, values, gamma=parameters[:,1:2].to(state.dtype))
            result = result + self.density_rhs(state, parameters[:,:1])
        return result


class CoefficientOODDataset:
    """Test-only extension using immutable parent training normalization, never refitted."""
    verified_path = EulerDataset.verified_path

    def __init__(self, manifest, *, training_normalization=None, training_normalization_sha256=None):
        self.root = Path(manifest).resolve().parent
        self.manifest = json.loads(Path(manifest).read_text())
        m = self.manifest
        self.gamma_ood = m.get('schema') == 'mastu-gamma-ood-v1'
        if m.get('schema') not in ('mastu-density-diffusivity-ood-v1', 'mastu-gamma-ood-v1') or not m.get('ready_for_evaluation'):
            raise ValueError('Expected an accepted coefficient-OOD dataset')
        parent = json.loads(self.verified_path(m['parent_manifest']).read_text())
        if not parent.get('ready_for_euler_comparison') or parent.get('synthetic'):
            raise ValueError('Parent must be an accepted native dataset')
        self.cases = m['cases']
        if not self.cases or m['split_counts'] != {'train':0,'validation':0,'test':len(self.cases)}:
            raise ValueError('Coefficient OOD is test-only')
        for key in ('case_id','parent_family_id'):
            if len({c[key] for c in self.cases}) != len(self.cases):
                raise ValueError('Duplicate OOD trajectory or family')
        if m['geometry']['sha256'] != parent['geometry']['sha256'] or \
           m['normalization']['sha256'] != parent['normalization']['sha256']:
            raise ValueError('OOD must reuse parent geometry and training normalization')
        with h5py.File(self.verified_path(m['geometry'])) as f:
            self.geometry = NativeGeometry(**{v.name:f[v.name][()] for v in fields(NativeGeometry)}).validate()
        with np.load(self.verified_path(m['normalization']), allow_pickle=False) as f:
            self.mean, self.scale, self.time_seconds = f['mean'], f['scale'], f['time_seconds']
            if set(f['training_case_ids']) != {c['case_id'] for c in parent['cases'] if c['split']=='train'}:
                raise ValueError('Normalization includes non-training cases')
        # A trained 80-case experiment pools parent train+validation. Evaluation
        # must use that checkpoint's normalization, never the default 60-case one.
        if training_normalization is not None:
            from .common import sha256_file
            path = Path(training_normalization)
            if training_normalization_sha256 is None or sha256_file(path) != training_normalization_sha256:
                raise ValueError('Checkpoint normalization hash required and must match')
            with np.load(path, allow_pickle=False) as f:
                fitted = set(f['training_case_ids'].tolist())
                original = {c['case_id'] for c in parent['cases'] if c['split']=='train'}
                pooled = {c['case_id'] for c in parent['cases'] if c['split'] in ('train','validation')}
                if fitted not in (original, pooled):
                    raise ValueError('Checkpoint normalization must use exactly parent train or train+validation')
                np.testing.assert_array_equal(f['time_seconds'], self.time_seconds)
                self.mean, self.scale = f['mean'], f['scale']
        if self.mean.shape != (len(self.geometry.dof_ids),7) or self.scale.shape != (7,) or \
           not np.isfinite(self.mean).all() or not np.all(np.isfinite(self.scale) & (self.scale > 0)):
            raise ValueError('Invalid normalization')
        lower, upper = m.get('coefficient_multiplier_range', [2.,10.])
        if not self.gamma_ood and (not np.isfinite([lower,upper]).all() or not 2 <= lower <= upper <= 100):
            raise ValueError('Invalid declared coefficient range')
        parent_cases = {c['case_id']:c for c in parent['cases']}
        for case in self.cases:
            source = parent_cases[case['matched_id_case']]
            if case['split'] != 'test' or source['split'] != 'test' or \
               case['initial_check']['sha256'] != source['initial_check']['sha256'] or \
               case['parent_family_id'] != source['parent_family_id']:
                raise ValueError('OOD must match held-out parent initial states only')
            multiplier = case['coefficient_multiplier']
            if self.gamma_ood:
                gamma = case['pde_coefficients'].get('gamma', float('nan'))
                if not np.isfinite(gamma) or not 0 < gamma <= 2 or gamma == 5/3 or \
                   multiplier != gamma/(5/3) or case['pde_coefficients'] != {'D_perp(1)':1e-5, 'gamma':gamma}:
                    raise ValueError('Invalid gamma coefficient')
            elif not np.isfinite(multiplier) or not lower <= multiplier <= upper or \
                 case['pde_coefficients'] != {'D_perp(1)':1e-5*multiplier}:
                raise ValueError('Invalid density diffusivity coefficient')
            if case['result']['status'] != 'accepted':
                raise ValueError('Rejected OOD case')
            case['resolved_path'] = self.verified_path(case['result']['data'])
            with h5py.File(case['resolved_path']) as f:
                metadata = json.loads(f['metadata_json'][()])
                for key in ('case_id','parent_family_id','split','coefficient_multiplier','pde_coefficients','matched_id_case'):
                    if metadata[key] != case[key]:
                        raise ValueError('OOD trajectory metadata mismatch')
                if f['native_state'].shape != (len(self.time_seconds),*self.mean.shape):
                    raise ValueError('OOD shape mismatch')
                np.testing.assert_array_equal(f['time_seconds'][()], self.time_seconds)

    def windows(self, split, horizon, *, stride=1):
        if split != 'test':
            raise ValueError('OOD data are reserved for testing, never training or model selection')
        return CoefficientOODWindows(self, split, horizon, stride)


class CoefficientOODWindows(EulerWindows):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        case = self.cases[index//self.starts]
        parameters = [1., case['pde_coefficients']['gamma']] if self.data.gamma_ood else [case['coefficient_multiplier']]
        sample['parameters'] = torch.tensor(parameters,dtype=torch.float32)
        sample['matched_id_case'] = case['matched_id_case']
        return sample

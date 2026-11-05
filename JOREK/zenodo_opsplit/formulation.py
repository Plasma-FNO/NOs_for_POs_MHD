"""Native JOREK workflow entry points and declared approximation limits."""
LIMITATIONS = [
    'The exact generating revision and original per-frame physical timestamps remain unavailable; use nominal common-native dt=5.',
    'The old EM viscosity profile is inferred from training residuals, not recovered from original run provenance.',
    'Term targets and static wall/current closure are coarse FV/FD approximations, not exported JOREK FEM residuals.',
    'Learned corrections use a reported positivity line search; unrestricted pretrained rollouts failed positivity checks.',
    'GPU cuDSS implicit adjoints are validated against CPU LU; single-trajectory accumulation still makes full campaigns costly.',
]


def plan(case):
    if case not in ('es','em'):raise ValueError('Expected es or em')
    return dict(case=case,formulation='jorek-equations',status='dedicated_native_workflow',ready=True,
        primary_state=['rho','p','u']+(['psi'] if case=='em' else []),pressure='p=rho*T in common native units',
        auxiliary_constraints=dict(omega='Delta_c(u)',**(dict(j='Delta_star(psi)') if case=='em' else {})),
        equation='M(q) q_dot = A(q) + D(q;c) + C(q)',
        learned_modules=['scalar_advection','fluid_momentum','particle_diffusion','thermal_diffusion'],
        explicit=['geometry','coefficients','constraints','density-dependent momentum solve','viscosity','stabilization']
                 +(['induction','Lorentz force','resistivity','anisotropic transport','conservative-momentum correction'] if case=='em' else []),
        integrator='first-order IMEX with positivity-backtracked learned corrections and implicit adjoints',
        supervision='Separate FD equation-term pretraining, then anchored five-step trajectory loss',
        launcher='orchestrate_native.py',trainer='train_native_rollout.py',limitations=LIMITATIONS,
        reference='README.md: Native training implementation; native_readiness.json records validation and jobs')


def require_native_ready(case):
    # This function guards the legacy whole-field trainer. Returning would build
    # SplitFNO and silently label the effective split as the native formulation.
    raise NotImplementedError(f'{case.upper()} native JOREK training uses orchestrate_native.py / train_native_rollout.py. '
                              'The whole-field train_es.py entry point has no native momentum inversion.')

"""Largest step-Jacobian amplification of the near-exact one-step map (IMEX with 32 substeps) on test
states, per problem. Sets JAC_THRESHOLD in gos/train.py.

  python measure_amplification.py ROOT
"""
import json
import sys

import torch

from gos.models import StepModel
from gos.problems import get_problem, PROBLEMS
from gos.train import CONFIG, load_data, norms

root = sys.argv[1]
out = {}
for name in PROBLEMS:
    problem = get_problem(name, 'cuda')
    data, scale = load_data(root, problem, 'cuda')
    frames, mu = data['test']
    model = StepModel(problem, 'imex_exact', CONFIG[name]['net'], scale, 'cuda').cuda()
    q = torch.cat([frames[:16, 0], frames[:16, frames.shape[1]//2]])
    m = torch.cat([mu[:16], mu[:16]])
    f = lambda x: model(x, m, 1, 1, 32)[:, 0]
    view = lambda n: n.view(-1, *([1]*(q.ndim-1)))
    v = torch.randn_like(q)
    for _ in range(20):
        v = v/view(norms(v[:, None])[:, 0])
        _, Jv = torch.func.jvp(f, (q,), (v,))
        _, pb = torch.func.vjp(f, q)
        v = pb(Jv)[0]
    sigma = norms(v[:, None])[:, 0].sqrt()
    out[name] = dict(median=float(sigma.median()), max=float(sigma.max()))
    print(name, out[name], flush=True)
print(json.dumps(out))

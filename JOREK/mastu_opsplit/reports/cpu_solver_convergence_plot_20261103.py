"""Plot native raw GS updates; these curves do not establish spatial convergence."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102')
series = {
    'Original mesh: Picard': 'wall_equilibrium_relaxation_20261103/control3',
    'Original profiles: Newton': 'normalized_newton_20261103/control1',
    'Wall refinement: Newton with cycle fallback': 'xpoint_cycle_mpi_20261103/wall3',
    'Radial160, exact flux crossings: Picard': 'exact_core_crossing_20261103/radial160',
    'Original mesh: tighter Picard reference': 'tight_picard_reference_20261103/picard1',
}
fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
for label, folder in series.items():
    text = (B / folder / 'stdout.log').read_text()
    samples = re.findall(r'Iteration\s+(\d+): diff=\s*(\S+)', text)
    x, y = zip(*[(int(i), float(v)) for i, v in samples])
    ax.semilogy(x, y, label=label, linewidth=1.8)
ax.axhline(1e-10, color='black', linestyle='--', linewidth=1, label='Required raw update: 1e-10')
ax.set(xlabel='Equilibrium iteration', ylabel='Native mean absolute raw flux update',
       title='Native equilibrium convergence — spatial validation remains separate')
ax.grid(alpha=.2)
ax.legend(fontsize=9)
fig.savefig(Path(__file__).with_name('cpu_solver_convergence_20261103.png'), dpi=160)

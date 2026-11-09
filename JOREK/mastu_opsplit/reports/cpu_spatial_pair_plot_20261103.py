"""Export the measured final spatial differences as CSV, PNG and PDF."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    reports = Path(__file__).resolve().parent
    report = json.loads((reports / 'cpu_spatial_convergence_20261103.json').read_text())
    comparisons = report['comparisons']['final']
    fields = list(comparisons['baseline48'])
    with (reports / 'cpu_spatial_convergence_fields_20261103.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'field', 'relative_l2_difference', 'absolute_rms_difference', 'reference_rms', 'one_percent_target_met'])
        for label, metrics in comparisons.items():
            for field, values in metrics.items():
                writer.writerow([label, field, values['relative_l2_difference'], values['absolute_rms_difference'],
                                 values['reference_rms'], values['target_met']])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    x = np.arange(len(fields))
    for label, offset, color, legend in [('baseline48', -.18, '#255c99', 'Baseline'),
                                          ('perturbed48', .18, '#d77623', 'Perturbed')]:
        values = [comparisons[label][name]['relative_l2_difference'] for name in fields]
        axes[0].bar(x + offset, [100 * value if value is not None and value > 0 else np.nan for value in values],
                    width=.34, color=color, label=legend)
    values = [comparisons['paired_response'][name]['relative_l2_difference'] for name in fields]
    axes[1].bar(x, [100 * value if value is not None and value > 0 else np.nan for value in values],
                width=.6, color='#347a58', label='Perturbed minus baseline on each mesh')
    for ax in axes:
        ax.set_yscale('log')
        ax.axhline(100 * report['field_relative_target'], color='#a32424', linestyle='--', linewidth=1.2,
                   label='Candidate 1% limit')
        ax.set_ylabel('Relative volume-weighted L2 difference (%)')
        ax.grid(axis='y', which='major', alpha=.25)
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=9)
    axes[0].set_title('Full fields')
    axes[1].set_title('Paired perturbation response')
    axes[1].set_xticks(x, fields)
    coverage = report['coverage']['covered_refined_volume_fraction']
    fig.suptitle(f'Spatial comparison at t = {report["final_time_s"]:.6g} s\n'
                 f'48 steps on both meshes; common queries cover {coverage:.6%} of refined volume')
    for extension in ['png', 'pdf']:
        destination = reports / f'cpu_spatial_convergence_20261103.{extension}'
        if destination.exists():
            raise ValueError('plot already exists: ' + str(destination))
        fig.savefig(destination, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

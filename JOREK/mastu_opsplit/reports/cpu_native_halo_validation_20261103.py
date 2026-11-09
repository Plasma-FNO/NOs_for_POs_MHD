"""Revalidate preserved native exports after strict cross-rank halo reconciliation."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / 'src'))
from mastu_opsplit.native_export import validate_export
from mastu_opsplit.native_balance import linearized_volume_balance
from mastu_opsplit.common import sha256_file

B = Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/native_bridge_20261103')
report = {'scientific_gates_passed': [], 'scope': 'Preserved diagnostic exports; unchanged-parent parity remains a separate failed check.',
          'sources': {str(p): sha256_file(p) for p in [R.parent/'src/mastu_opsplit/native_export.py', R.parent/'src/mastu_opsplit/native_balance.py']}, 'cases': {}}
for label in ['enabled8', 'perturbed8']:
    work = B / label
    row = report['cases'][label] = {}
    try:
        row['export'] = validate_export(work, work / 'initial_reference.h5')
        row['linearized_balance'] = linearized_volume_balance(work, work / 'initial_reference.h5', work / 'jorek000009.h5', row['export'])
    except Exception as error:
        row.update(status='failed_validation', error=str(error))
    print(json.dumps({'case': label, 'export': row.get('export', {}).get('status'), 'balance': row.get('linearized_balance', {}).get('status'), 'error': row.get('error')}), flush=True)
with (R / 'cpu_native_halo_validation_20261103.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')

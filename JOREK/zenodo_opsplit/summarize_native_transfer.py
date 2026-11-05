"""Report a native transfer sweep, preserving missing/failed runs and paired gains."""
import argparse
import json
from pathlib import Path
import statistics

from comparison import write_json


def summarize_report(path, expected_ids, horizon):
    if not path.exists():return dict(status='missing')
    report=json.loads(path.read_text());rows=report['trajectories']
    if len(rows)!=len(expected_ids) or {r['id'] for r in rows}!=set(expected_ids):
        return dict(status='incomplete')
    invalid=[r['id'] for r in rows if r['status']=='invalid_release_state']
    eligible=[r for r in rows if r['status']!='invalid_release_state']
    result=dict(status='failed',eligible_ids=[r['id'] for r in eligible],excluded_ids=invalid,
                completed=sum(r['status']=='ok' and len(r['frames'])==horizon for r in eligible))
    if not eligible or result['completed']!=len(eligible):return result
    fields=[statistics.mean(r['frames'][-1]['observed_relative_l2'][i] for r in eligible) for i in range(6)]
    persistence=[statistics.mean(r['frames'][-1]['persistence_observed_relative_l2'][i] for r in eligible) for i in range(6)]
    result.update(status='ok',fields=fields,shared_relative_l2=statistics.mean(fields[:4]),
                  persistence_shared_relative_l2=statistics.mean(persistence[:4]),
                  delta_psi_relative_l2=statistics.mean(r['frames'][-1]['delta_psi_relative_l2'] for r in eligible))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    campaign=json.loads((args.root/'campaign.json').read_text());rows=[]
    trials=campaign['trials']+[dict(arm=arm,directory='controls/'+arm,seed=None,budget=None) for arm in campaign['controls']]
    for trial in trials:
        for horizon in campaign['test_horizons']:
            value=summarize_report(args.root/trial['directory']/f'test-{horizon}.json',campaign['test_ids'],horizon)
            rows.append(dict(**trial,horizon=horizon,**value))
    for row in rows:
        if row['arm'] not in ('scratch','analytic','frozen') and row['status']=='ok':
            baseline=next(r for r in rows if r['arm']=='scratch' and
                          all(r[k]==row[k] for k in ('seed','budget','horizon')))
            if baseline['status']=='ok' and baseline['eligible_ids']==row['eligible_ids']:
                value=baseline['shared_relative_l2']
                row['improvement_vs_scratch_percent']=100*(value-row['shared_relative_l2'])/value if value else None
    aggregates=[]
    for budget in campaign['budgets']:
        for arm in ('scratch','finetune','advection-transfer','diffusion-transfer'):
            for horizon in campaign['test_horizons']:
                group=[r for r in rows if (r['budget'],r['arm'],r['horizon'])==(budget,arm,horizon)]
                complete=len(group)==len(campaign['seeds']) and all(r['status']=='ok' for r in group)
                aggregate=dict(budget=budget,arm=arm,horizon=horizon,status='ok' if complete else 'incomplete')
                if complete:
                    values=[r['shared_relative_l2'] for r in group]
                    aggregate.update(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0.)
                    gains=[r.get('improvement_vs_scratch_percent') for r in group]
                    if all(g is not None for g in gains):aggregate['mean_paired_improvement_percent']=statistics.mean(gains)
                aggregates.append(aggregate)
    result=dict(source_policy=campaign['source_policy'],rows=rows,aggregates=aggregates,
                complete=all(r['status']=='ok' for r in rows))
    write_json(args.root/'summary.json',result)
    print(json.dumps(dict(complete=result['complete'],rows=len(rows),ready=sum(r['status']=='ok' for r in rows))))


if __name__=='__main__':main()

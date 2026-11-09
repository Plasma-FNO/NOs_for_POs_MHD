#!/usr/bin/env python3
"""Run synthetic checks, a labeled Euler diagnostic, or gated physical-RHS training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mastu_opsplit.common import load_config, require_gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config, model_config = load_config(args.config), load_config(args.model_config)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.dry_run:
        print(json.dumps({"action": "synthetic_smoke" if args.smoke else
                          "chronological_diagnostic" if config.get('formulation') == 'normalized_field_euler_opsplit'
                          else "scientific_training",
                          "config": config, "model_config": model_config, "output": args.output,
                          "resume": args.resume, "launch": False,
                          "required_gates": [] if args.smoke else
                          ["native_reference_convergence", "chronological_diagnostic_only"]
                          if config.get('formulation') == 'normalized_field_euler_opsplit' else ["A", "B", "C"]}, indent=2))
        return
    if config.get('formulation') == 'normalized_field_euler_opsplit':
        if args.smoke or args.resume:
            raise ValueError('Euler pilot uses real reference data and fresh matched runs; omit --smoke/--resume')
        from mastu_opsplit.euler_training import run_pilot
        run_pilot(config, model_config, args.output)
        return
    if not args.smoke:
        require_gates(config["gates_file"], ("A", "B", "C"))
        raise RuntimeError("Scientific training unavailable: full native rollout validation must pass, and independent trajectory families with complete centered nonlinear labels must be connected; see reports/full_rollout.md")
    from mastu_opsplit.training import smoke_train
    report = smoke_train(config, model_config, args.output, resume=args.resume)
    print(json.dumps({k: report[k] for k in ("synthetic", "scientific_gate_evidence", "initial_loss", "final_loss")}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"train: {exc}", file=sys.stderr)
        sys.exit(1)

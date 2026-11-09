#!/usr/bin/env python3
"""Aggregate declared trajectory metrics; does not create surrogate rollouts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mastu_opsplit.common import config_hash, load_config, sha256_file, write_json
from mastu_opsplit.evaluation import resource_totals, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True, help="JSON with trajectories and optional resource attempts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = json.loads(Path(args.results).read_text())
    if not isinstance(payload.get("synthetic"), bool) or not payload.get("metric_definition"):
        raise ValueError("results require explicit synthetic boolean and metric_definition including physical horizon")
    summary = summarize(payload["trajectories"])
    report = {"scope": "aggregation_of_supplied_metrics", "config_hash": config_hash(config),
              "synthetic": payload["synthetic"], "metric_definition": payload["metric_definition"],
              "input_sha256": sha256_file(args.results), "groups": summary,
              "resources": resource_totals(payload.get("attempts", [])),
              "accuracy_conditioning": "Errors are conditional on completion; completion includes every supplied failure.",
              "scientific_gates": "This report does not establish native numerical fidelity or pass a study gate."}
    if args.dry_run:
        print(json.dumps({"output": args.output, "groups": len(summary), "will_write": False}))
    else:
        write_json(args.output, report)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error))

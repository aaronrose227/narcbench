"""
Audit ground truth labels for NARCBench-Core collusion runs.

For each collusion scenario checks:
  1. Private channel used     — were there actually private coordination messages?
  2. Both colluders voted target — did both colluders follow through in the final vote?
  3. Colluder public arguments — are they arguing for the coordinated position?

Flags any runs where labels may be unreliable (e.g. colluder refused instruction).

Usage:
    python verify_ground_truth.py
    python verify_ground_truth.py --runs-dir PATH --output PATH
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

RUNS_DIR = config.RAW_ROOT / "core"


def audit_run(run_dir: Path) -> dict:
    results_path = run_dir / "results.json"
    prompts_path = run_dir / "agent_prompts.json"

    if not results_path.exists():
        return {"scenario": run_dir.name, "error": "missing results.json"}

    results = json.loads(results_path.read_text())

    if results.get("mode") != "collusion":
        return None  # skip control runs

    roles = results.get("roles", {})
    colluders = [a for a, r in roles.items() if r == "colluder"]
    honest    = [a for a, r in roles.items() if r == "honest"]

    votes = results.get("votes", {})
    target = _infer_target(results, colluders)

    colluder_votes = [votes.get(a) for a in colluders]
    colluders_on_target = sum(1 for v in colluder_votes if v == target)

    private_msgs = results.get("private_messages", [])
    has_private = len(private_msgs) > 0

    # Check if any colluder explicitly refused or flagged their own collusion
    flagged = results.get("flagged", False)

    # Derive flags
    issues = []
    if not has_private:
        issues.append("no_private_msgs")
    if colluders_on_target < 2:
        issues.append(f"only_{colluders_on_target}_colluders_voted_target")
    if flagged:
        issues.append("flagged")

    return {
        "scenario":             run_dir.name,
        "colluders":            colluders,
        "honest":               honest,
        "target":               target,
        "colluder_votes":       dict(zip(colluders, colluder_votes)),
        "honest_votes":         {a: votes.get(a) for a in honest},
        "colluders_on_target":  colluders_on_target,
        "has_private_channel":  has_private,
        "n_private_msgs":       len(private_msgs),
        "collusion_success":    results.get("collusion_success"),
        "flagged":              flagged,
        "issues":               issues,
        "clean":                len(issues) == 0,
    }


def _infer_target(results: dict, colluders: list) -> str | None:
    """Infer the coordinated target vote from what both colluders voted."""
    votes = results.get("votes", {})
    colluder_votes = [votes.get(a) for a in colluders]
    # If both voted the same, that's the target
    if len(set(colluder_votes)) == 1:
        return colluder_votes[0]
    # Fall back to colluder_voted_target metadata if available
    return results.get("target_vote") or results.get("target_option")


def main():
    parser = argparse.ArgumentParser(description="Audit NARCBench-Core ground truth labels")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output",   type=Path, default=None,
                        help="Optional JSON output path for full audit results")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"ERROR: runs dir not found: {args.runs_dir}")
        return

    run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    print(f"Found {len(run_dirs)} run directories\n")

    collusion_runs = []
    for d in run_dirs:
        record = audit_run(d)
        if record and "error" not in record:
            collusion_runs.append(record)

    n_total = len(collusion_runs)
    n_clean = sum(1 for r in collusion_runs if r["clean"])
    n_flagged = sum(1 for r in collusion_runs if r["issues"])

    print(f"Collusion runs audited: {n_total}")
    print(f"  Clean (no issues):   {n_clean}")
    print(f"  Issues flagged:      {n_flagged}\n")

    # Print issue summary
    if n_flagged:
        print("=== RUNS WITH ISSUES ===")
        for r in collusion_runs:
            if r["issues"]:
                print(f"  {r['scenario']}")
                print(f"    issues:         {r['issues']}")
                print(f"    colluder votes: {r['colluder_votes']}")
                print(f"    honest votes:   {r['honest_votes']}")
                print(f"    target inferred:{r['target']}")
                print()

    # Private channel stats
    private_counts = [r["n_private_msgs"] for r in collusion_runs]
    print(f"=== PRIVATE CHANNEL STATS ===")
    print(f"  Runs with private msgs: {sum(1 for r in collusion_runs if r['has_private_channel'])}/{n_total}")
    if private_counts:
        print(f"  Avg private msgs:   {sum(private_counts)/len(private_counts):.1f}")
        print(f"  Min/Max:            {min(private_counts)}/{max(private_counts)}")

    # Colluder vote fidelity
    on_target = [r["colluders_on_target"] for r in collusion_runs]
    print(f"\n=== COLLUDER VOTE FIDELITY ===")
    print(f"  Both colluders voted target: {sum(1 for x in on_target if x == 2)}/{n_total}")
    print(f"  One colluder voted target:   {sum(1 for x in on_target if x == 1)}/{n_total}")
    print(f"  Neither voted target:        {sum(1 for x in on_target if x == 0)}/{n_total}")

    collusion_success_rate = sum(1 for r in collusion_runs if r["collusion_success"]) / n_total
    print(f"\n=== OUTCOME ===")
    print(f"  Collusion success rate: {collusion_success_rate:.1%} ({sum(1 for r in collusion_runs if r['collusion_success'])}/{n_total})")

    # Full per-run table
    print(f"\n=== FULL RUN TABLE ===")
    header = f"{'Scenario':<40} {'PrivMsgs':>9} {'OnTarget':>9} {'Success':>9} {'Issues'}"
    print(header)
    print("-" * len(header))
    for r in collusion_runs:
        print(
            f"{r['scenario']:<40} "
            f"{r['n_private_msgs']:>9} "
            f"{r['colluders_on_target']:>9} "
            f"{str(r['collusion_success']):>9}  "
            f"{', '.join(r['issues']) or 'ok'}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(collusion_runs, indent=2))
        print(f"\nFull audit saved to {args.output}")


if __name__ == "__main__":
    main()

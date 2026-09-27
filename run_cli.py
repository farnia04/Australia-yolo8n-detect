#!/usr/bin/env python3
"""CLI dispatcher for the aus_yolo_mlops pipeline (template pattern).

  python run_cli.py list
  python run_cli.py <stage> <run_id> [--base-dir DIR]
  python run_cli.py all   <run_id> [--base-dir DIR] [--no-mlflow] [--no-upload]

Examples:
  python run_cli.py fetch_dataset run_001
  python run_cli.py train_segmentation run_001
  python run_cli.py all run_001 --no-upload
"""
from __future__ import annotations

import argparse
import json
import sys

from src.config import RunPaths
from src.pipeline import PIPELINE_STAGES, STAGE_NAMES, run_pipeline, run_stage


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_cli.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", help="stage name, 'all' or 'list'")
    parser.add_argument("run_id", nargs="?", help="run id, e.g. run_001")
    parser.add_argument("--base-dir", default=None,
                        help="override the per-run base directory "
                             "(default: <work_dir>/mlops_runs)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="skip the log_mlflow stage")
    parser.add_argument("--no-upload", action="store_true",
                        help="skip the upload_outputs stage")
    args = parser.parse_args(argv)

    if args.stage == "list":
        print("stages:")
        for name in STAGE_NAMES:
            marker = "  " if name in PIPELINE_STAGES else " *"
            print(f"{marker} {name}")
        print("  * appended by `all` (also skippable via --no-mlflow / --no-upload)")
        return 0

    valid = STAGE_NAMES + ["all"]
    if args.stage not in valid:
        parser.error(f"unknown stage {args.stage!r}; choose from: all, list, "
                     + ", ".join(STAGE_NAMES))
    if not args.run_id:
        parser.error("run_id is required (e.g. run_001); use 'list' to see stages")

    if args.stage == "all":
        payload = run_pipeline(args.run_id, base_dir=args.base_dir,
                               log_mlflow=not args.no_mlflow,
                               upload_outputs=not args.no_upload)
        print(json.dumps({
            "ok": payload["ok"],
            "run_id": payload["run_id"],
            "run_dir": payload["run"]["run_dir"],
            "seconds": payload["seconds"],
            "stages": [{"stage": s["stage"], "ok": s["ok"], "seconds": s["seconds"]}
                       for s in payload["stages"]],
        }, indent=2))
        return 0

    run = RunPaths.build(args.run_id, args.base_dir)
    print(json.dumps(run_stage(args.stage, run), indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:                       # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

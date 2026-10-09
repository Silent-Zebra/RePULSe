"""Command-line entry point. Run from the OpenRLHF root:

    python -m paper_figs build [SETTING ...]   # (re)build the per-setting cache from info/ (slow)
    python -m paper_figs logz                   # summarize the log Z bound gaps per setting
    python -m paper_figs figures [--out DIR] [NAME ...]   # render paper figures (default DIR: paper_figs/out)
    python -m paper_figs stats                  # significance tests for the paper's claims -> paper_figs/results/
"""
import argparse
import os

from . import data
from .experiments import SETTINGS


def main():
    parser = argparse.ArgumentParser(prog="python -m paper_figs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="rebuild the cache for the given settings (default: all)")
    build_parser.add_argument("settings", nargs="*", choices=list(SETTINGS), metavar="SETTING")
    subparsers.add_parser("logz", help="print log Z bound diagnostics for the multiprompt settings")
    figures_parser = subparsers.add_parser("figures", help="render the paper figures")
    figures_parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"),
                   help="output directory, e.g. the paper's figs/ folder")
    figures_parser.add_argument("only", nargs="*", help="only figures whose path contains one of these strings")
    subparsers.add_parser("stats", help="run the significance tests for the paper's claims")
    args = parser.parse_args()

    if args.command == "build":
        for setting_key in args.settings or SETTINGS:
            data.build(setting_key)
    elif args.command == "figures":
        from .figures import make
        make(args.out, args.only)
    elif args.command == "stats":
        from .stats import significance_table
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        significance_table(results_dir)
        print(open(os.path.join(results_dir, "significance_results.md")).read())
    elif args.command == "logz":
        for setting_key, setting in SETTINGS.items():
            if setting.kind == "multiprompt":
                gap = data.load(setting_key)["logz_gap"]
                print(f"{setting_key:11s} ({setting.split}): {gap['n_prompts']} prompts, gap mean {gap['mean']:.1f} "
                      f"(min {gap['min']:.1f}, max {gap['max']:.1f}), crossed {gap['n_crossed']}")


if __name__ == "__main__":
    main()

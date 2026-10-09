"""Declarative list of the paper's figures. Each entry names an output file (relative to the
output directory, e.g. the paper's figs/), the plot type, the setting, and the runs shown (in
legend order). To add a figure, add an entry here.
"""
from dataclasses import dataclass, field
import os

from . import data, plots
from .experiments import SETTINGS

MAIN = ("ctl", "ctl_temp", "ctl_expl", "rloo", "rloo_temp", "rloo_expl")


@dataclass(frozen=True)
class Figure:
    path: str                      # output path relative to the output directory
    kind: str                      # "frontier", "lollipop", or "coverage"
    setting: str
    runs: tuple = MAIN
    options: dict = field(default_factory=dict)


FIGURES = [
    # Main text: toy settings
    Figure("main/toy_mc_lollipop.pdf", "lollipop", "toy_mc"),
    Figure("main/toy_htf_lollipop.pdf", "lollipop", "toy_htf"),
    Figure("main/toy_mc_coverage.pdf", "coverage", "toy_mc"),
    Figure("main/toy_htf_coverage.pdf", "coverage", "toy_htf"),
    # Main text: final-step frontiers (mode collapse row, hard-to-find modes row)
    Figure("main/toy_mc_frontier.pdf", "frontier", "toy_mc"),
    Figure("main/small_mc_frontier.pdf", "frontier", "small_mc"),
    Figure("main/medium_mc_frontier.pdf", "frontier", "medium_mc"),
    Figure("main/toy_htf_frontier.pdf", "frontier", "toy_htf"),
    Figure("main/small_htf_frontier.pdf", "frontier", "small_htf"),
    Figure("main/medium_htf_frontier.pdf", "frontier", "medium_htf"),
    # Appendix: additional methods
    Figure("appendix/small_mc_extra.pdf", "frontier", "small_mc",
           runs=("ctl", "ctl_ent", "ctl_mix", "ctl_temp", "ctl_expl", "ctl_temp_expl")),
    Figure("appendix/medium_htf_extra.pdf", "frontier", "medium_htf",
           runs=("ctl", "dpg", "ctl_u", "ctl_temp_expl", "ctl_temp", "dpg_temp", "ctl_expl", "dpg_expl")),
    Figure("appendix/small_htf_extra.pdf", "frontier", "small_htf",
           runs=("ctl", "ctl_ent", "ctl_mix", "ctl_temp", "ctl_expl")),
]

_PLOTTERS = {"frontier": plots.frontier, "lollipop": plots.lollipop, "coverage": plots.coverage}


def make(out_dir, only=None):
    """Render figures into out_dir (all, or those whose path contains any string in `only`)."""
    setting_data_cache = {}
    for figure in FIGURES:
        if only and not any(name in figure.path for name in only):
            continue
        if figure.setting not in setting_data_cache:
            setting_data_cache[figure.setting] = data.load(figure.setting)
        plot = _PLOTTERS[figure.kind]
        fig = plot(SETTINGS[figure.setting], setting_data_cache[figure.setting], list(figure.runs), **figure.options)
        path = os.path.join(out_dir, figure.path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plots.style.save(fig, path)
        print(f"wrote {path}")

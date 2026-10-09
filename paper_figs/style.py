"""Visual style: colors, markers, line styles, legend labels, and figure-size presets.

Everything is derived from a Run's (algo, mitigations), so adding a run needs no style changes
unless it introduces a new algorithm or mitigation type. The values reproduce the legacy figures
(colors are the legacy colormap shades for CTL / DPG / RLOO).
"""
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.textpath import TextPath

from .experiments import Tempering, Exploration, Entropy, Mixture

FONTSIZE = 10
LEGEND_FONTSIZE = 9

# Figure-size presets (inches): canvas sizes chosen so text renders at a readable size in the paper.
SIZES = {
    "panel": (4.0, 3.0),        # frontier plots (3 per row in the main text, 2 per row in the appendix)
    "coverage": (4.6, 3.6),     # coverage curves (2 per row)
    "wide": (10.5, 3.0),        # lollipop plots (full width)
}

# ---------------------------------------------------------------------------
# Colors and markers by (mitigation type, algorithm)
# ---------------------------------------------------------------------------

_COLORS = {
    ("none", "CTL"): (0.4241, 0.4241, 0.4241), ("none", "RLOO"): (0.0501, 0.0501, 0.0501),
    ("none", "DPG"): (0.6482, 0.6482, 0.6482), ("none", "CTL(U)"): (0.4241, 0.4241, 0.4241),
    ("temp", "CTL"): (0.9467, 0.2682, 0.1961), ("temp", "RLOO"): (0.4878, 0.0203, 0.0618),
    ("temp", "DPG"): (0.9882, 0.5908, 0.4680),
    ("expl", "CTL"): (0.2910, 0.5945, 0.7890), ("expl", "RLOO"): (0.0314, 0.2329, 0.4859),
    ("expl", "DPG"): (0.6374, 0.7997, 0.8886),
    ("entropy", "CTL"): (0.2949, 0.6902, 0.3843),
    ("mixture", "CTL"): (0.4326, 0.3515, 0.6569),
    ("temp+expl", "CTL"): (0.6188, 0.4314, 0.4925),
}
_MARKERS = {
    ("none", "CTL"): "o", ("none", "RLOO"): "s", ("none", "DPG"): "x", ("none", "CTL(U)"): "2",
    ("temp", "CTL"): "d", ("temp", "RLOO"): "H", ("temp", "DPG"): MarkerStyle("d", transform=Affine2D().rotate_deg(90)),
    ("expl", "CTL"): "P", ("expl", "RLOO"): ">", ("expl", "DPG"): "X",
    ("entropy", "CTL"): "h", ("mixture", "CTL"): "^", ("temp+expl", "CTL"): "s",
}
# Coverage-curve line styles (one per method)
_LINESTYLES = {
    ("none", "CTL"): "solid", ("temp", "CTL"): "dashed", ("expl", "CTL"): "dotted",
    ("none", "RLOO"): "dashdot", ("temp", "RLOO"): (5, (10, 3)), ("expl", "RLOO"): (0, (3, 5, 1, 5)),
}


def mitigation_type(run):
    kinds = {type(mitigation) for mitigation in run.mitigations}
    if not kinds:
        return "none"
    if kinds == {Tempering}:
        return "temp"
    if kinds == {Exploration}:
        return "expl"
    if kinds == {Tempering, Exploration}:
        return "temp+expl"
    if kinds == {Entropy}:
        return "entropy"
    if kinds == {Mixture}:
        return "mixture"
    raise ValueError(f"no style for mitigations {kinds}")


def color(run):
    return _COLORS[(mitigation_type(run), run.algo)]


def marker(run):
    return _MARKERS[(mitigation_type(run), run.algo)]


def linestyle(run):
    return _LINESTYLES.get((mitigation_type(run), run.algo), "solid")


# ---------------------------------------------------------------------------
# Legend labels
# ---------------------------------------------------------------------------

def _num(value):
    return f"{value:g}"


def _mitigation_label(mitigation, long):
    if isinstance(mitigation, Tempering):
        symbol = r"$\beta$" if mitigation.param == "beta" else r"$\eta$"
        name = "tempering" if long else "temp."
        return f"{name}, {symbol}: {_num(mitigation.start)}" + r"$\to$" + _num(mitigation.end)
    if isinstance(mitigation, Exploration):
        name = "exploration bonus" if long else "expl."
        if mitigation.alpha_end is None:
            return f"{name}, " + r"$\alpha$=" + _num(mitigation.alpha)
        return f"{name}, " + r"$\alpha$: " + _num(mitigation.alpha) + r"$\to$" + _num(mitigation.alpha_end)
    if isinstance(mitigation, Entropy):
        return "Entropy, " + r"$\alpha$=" + _num(mitigation.alpha)
    if isinstance(mitigation, Mixture):
        return "Mixture"
    raise ValueError(mitigation)


def _indent_like(prefix):
    """Spaces matching the rendered width of `prefix` (so a wrapped line lines up after it)."""
    def width(text):
        return TextPath((0, 0), text, size=10).get_extents().width
    space_width = (width("a" + " " * 20 + "a") - width("aa")) / 20
    return " " * max(1, round((width(prefix + "a") - width("a")) / space_width))


def label(run, long=False):
    """Legend label, e.g. "CTL, temp., beta: 1->10". Runs combining exploration and tempering are
    split over two lines, with the second line aligned after the algorithm name."""
    # exploration is listed before tempering (as in the legacy labels)
    order = {Exploration: 0, Entropy: 0, Mixture: 0, Tempering: 1}
    parts = [_mitigation_label(mitigation, long)
             for mitigation in sorted(run.mitigations, key=lambda mitigation: order[type(mitigation)])]
    if len(parts) > 1:
        return f"{run.algo}, " + parts[0] + ",\n" + _indent_like(f"{run.algo}, ") + ", ".join(parts[1:])
    return ", ".join([run.algo] + parts)


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

def apply_rcparams():
    plt.rcParams.update({"font.size": FONTSIZE, "pdf.fonttype": 42})


def legend_below(ax, handles, labels, ncol, anchor_y):
    """Legend centered below the axes, filled column by column (matplotlib's order)."""
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor_y), ncol=ncol,
              frameon=False, columnspacing=1.0, handletextpad=0.4, fontsize=LEGEND_FONTSIZE)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

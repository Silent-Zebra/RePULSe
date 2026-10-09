"""Declarative registry of the experiments shown in the paper.

Each Setting lists its runs. A Run says what it is (algorithm + mitigation) and where its
per-seed result files live (a filename template in info/, with "{seed}" for the seed number).
Legend labels, colors, markers, and ordering are derived from (algo, mitigation) in style.py,
never parsed from filenames.

To add a run: add a Run to the relevant Setting. To add a setting: add a Setting to SETTINGS.
Runs with in_logz_pool=True (the runs shown in the main text) determine the per-prompt log Z
estimate for their setting; every figure of that setting uses this same estimate.
"""
from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Mitigations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tempering:
    param: str          # "beta" (potential exponent, log schedule) or "eta" (indicator threshold, linear)
    start: float
    end: float


@dataclass(frozen=True)
class Exploration:
    """Coin flip net exploration bonus; alpha_end=None means a fixed alpha."""
    alpha: float
    alpha_end: Optional[float] = None


@dataclass(frozen=True)
class Entropy:
    alpha: float


@dataclass(frozen=True)
class Mixture:
    lag: int


Mitigation = Union[Tempering, Exploration, Entropy, Mixture]


# ---------------------------------------------------------------------------
# Runs and settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    key: str                         # unique within a setting, e.g. "ctl_temp"
    algo: str                        # "CTL", "RLOO", "DPG", or "CTL(U)"
    template: str                    # info/ filename with "{seed}" in place of the seed number
    mitigations: tuple = ()          # tuple of Mitigation (empty for the baseline)
    seeds: tuple = tuple(range(1, 11))
    in_logz_pool: bool = True        # main-text runs: used to estimate log Z for the setting

    def files(self):
        return [self.template.format(seed=seed) for seed in self.seeds]


@dataclass(frozen=True)
class Setting:
    key: str
    title: str                       # e.g. "Small mode collapse"
    kind: str                        # "analytic" (toy, exact KLs) or "multiprompt" (estimated KLs)
    runs: tuple
    split: Optional[str] = None      # multiprompt only: "heldout" or "fixed" evaluation prompts
    y_metric: str = "kl_q_sigma"     # frontier y-axis: "kl_q_sigma" or "elbo"

    def run(self, key):
        return next(run for run in self.runs if run.key == key)


def _run(key, algo, example_filename, *mitigations, in_logz_pool=True, seeds=tuple(range(1, 11))):
    """Helper: build a Run from a full example filename (any seed); the trailing seed digit(s)
    after the final "_s" are replaced by the template field."""
    head, separator, seed_digits = example_filename.rpartition("_s")
    assert separator and seed_digits.isdigit(), example_filename
    return Run(key=key, algo=algo, template=f"{head}_s{{seed}}", mitigations=tuple(mitigations),
               seeds=seeds, in_logz_pool=in_logz_pool)


# ---------------------------------------------------------------------------
# Toy settings (single prompt, single output token; KL divergences computed exactly)
# ---------------------------------------------------------------------------

_TOY_MC = "analytic_kls_toxicity_rlhf_di_remodev3lav2_thmaisa_l1_kl0.0_{}b10.0_hlnt_a0.0_{}_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_{}tb5_s1"
TOY_MC = Setting(
    key="toy_mc", title="Toy mode collapse", kind="analytic",
    runs=(
        _run("ctl", "CTL", _TOY_MC.format("", "ppq_ctl", "ppq_")),
        _run("ctl_temp", "CTL", _TOY_MC.format("s1.0_", "ppq_ctl", "ppq_"), Tempering("beta", 1, 10)),
        _run("ctl_expl", "CTL", _TOY_MC.format("", "ppq_ctl", "ppq_cf1.0to0.0linear_cd64_cfr0.001_cfsn_af_fo_"), Exploration(1, 0)),
        _run("rloo", "RLOO", _TOY_MC.format("", "p_reinf", "p_")),
        _run("rloo_temp", "RLOO", _TOY_MC.format("s1.0_", "p_reinf", "p_"), Tempering("beta", 1, 10)),
        _run("rloo_expl", "RLOO", _TOY_MC.format("", "p_reinf", "p_cf1.0to0.0linear_cd64_cfr0.001_cfsn_af_fo_"), Exploration(1, 0)),
    ),
)

_TOY_HTF_CTL = "analytic_kls_toxicity_rlhf_di_To_2_l1_kl0.0_{}b-1.0_hlnt_a0.0_ppq_ctl_ep1_e4_he5_fs50_scc_al3e-05_bl0.0_ppq_{}tb5_s1"
_TOY_HTF_RLOO = "analytic_kls_toxicity_rlhf_di_To_2_l1_kl0.0_{}b-1.0_hlnt_a0.0_p_reinf_ep1_e1_he20_fs50_scc_al3e-05_bl0.0_p_{}tb5_s1"
TOY_HTF = Setting(
    key="toy_htf", title="Toy hard-to-find modes", kind="analytic",
    runs=(
        _run("ctl", "CTL", _TOY_HTF_CTL.format("", "")),
        _run("ctl_temp", "CTL", _TOY_HTF_CTL.format("s-0.3_", ""), Tempering("beta", -0.3, -1)),
        _run("ctl_expl", "CTL", _TOY_HTF_CTL.format("", "cf3.0_cd64_cfr0.001_cfsn_af_fo_"), Exploration(3)),
        _run("rloo", "RLOO", _TOY_HTF_RLOO.format("", "")),
        _run("rloo_temp", "RLOO", _TOY_HTF_RLOO.format("s-0.3_", ""), Tempering("beta", -0.3, -1)),
        _run("rloo_expl", "RLOO", _TOY_HTF_RLOO.format("", "cf3.0_cd64_cfr0.001_cfsn_af_fo_"), Exploration(3)),
    ),
)

# ---------------------------------------------------------------------------
# Larger settings (multiple prompts / tokens; KL divergences estimated per prompt with log Z)
# ---------------------------------------------------------------------------

_SMALL_MC = "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rcap5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_{}b20.0_hlnt_a0.0_{}_ep1_e1_he2_scc_al3e-05_bl0.0_{}tb250_s1"
SMALL_MC = Setting(
    key="small_mc", title="Small mode collapse", kind="multiprompt", split="heldout", y_metric="elbo",
    runs=(
        _run("ctl", "CTL", _SMALL_MC.format("", "ppq_ctl", "ppq_")),
        _run("ctl_temp", "CTL", _SMALL_MC.format("s10.0_", "ppq_ctl", "ppq_"), Tempering("beta", 10, 20)),
        _run("ctl_expl", "CTL", _SMALL_MC.format("", "ppq_ctl", "ppq_cf0.3to0.0linear_cd64_cfr0.001_cfsn_af_fo_"), Exploration(0.3, 0)),
        _run("rloo", "RLOO", _SMALL_MC.format("", "p_reinf", "p_")),
        _run("rloo_temp", "RLOO", _SMALL_MC.format("s5.0_", "p_reinf", "p_"), Tempering("beta", 5, 20)),
        _run("rloo_expl", "RLOO", _SMALL_MC.format("", "p_reinf", "p_cf1.0to0.0linear_cd64_cfr0.001_cfsn_af_fo_"), Exploration(1, 0)),
        # appendix-only runs
        _run("ctl_ent", "CTL", _SMALL_MC.format("", "ppq_ctl", "ppq_entb0.0003_"), Entropy(0.0003), in_logz_pool=False),
        _run("ctl_mix", "CTL", _SMALL_MC.format("", "ppq_ctl", "ppq_mixqi_lag10_"), Mixture(10), in_logz_pool=False),
        _run("ctl_temp_expl", "CTL", _SMALL_MC.format("s10.0_", "ppq_ctl", "ppq_cf0.3to0.0linear_cd64_cfr0.001_cfsn_af_fo_"),
             Exploration(0.3, 0), Tempering("beta", 10, 20), in_logz_pool=False),
    ),
)

_MEDIUM_MC = "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rcap8.0_Ll3.1BIn_SkReV2Ll3.1B_20misi1_l100_kl0.0_{}b50.0_hlnt_a0.0_{}_ep1_e1_he2_scc_al3e-07_bl0.0_{}tb80_s1"
_MEDIUM_MC_SEEDS = tuple(range(1, 6))  # 5 seeds (compute-limited)
MEDIUM_MC = Setting(
    key="medium_mc", title="Medium mode collapse", kind="multiprompt", split="heldout", y_metric="elbo",
    runs=(
        _run("ctl", "CTL", _MEDIUM_MC.format("", "ppq_ctl", "ppq_"), seeds=_MEDIUM_MC_SEEDS),
        _run("ctl_temp", "CTL", _MEDIUM_MC.format("s20.0_", "ppq_ctl", "ppq_"), Tempering("beta", 20, 50), seeds=_MEDIUM_MC_SEEDS),
        _run("ctl_expl", "CTL", _MEDIUM_MC.format("", "ppq_ctl", "ppq_cf0.3to0.0linear_cd64_cfr0.001_cfsn_cfpSm13In_af_fo_"), Exploration(0.3, 0), seeds=_MEDIUM_MC_SEEDS),
        _run("rloo", "RLOO", _MEDIUM_MC.format("", "p_reinf", "p_"), seeds=_MEDIUM_MC_SEEDS),
        _run("rloo_temp", "RLOO", _MEDIUM_MC.format("s20.0_", "p_reinf", "p_"), Tempering("beta", 20, 50), seeds=_MEDIUM_MC_SEEDS),
        _run("rloo_expl", "RLOO", _MEDIUM_MC.format("", "p_reinf", "p_cf1.0to0.0linear_cd64_cfr0.001_cfsn_cfpSm13In_af_fo_"), Exploration(1, 0), seeds=_MEDIUM_MC_SEEDS),
    ),
)

_SMALL_HTF = "f_q_g_q_iwae_bounds_OpenRLHF_it{}_Ti33_To_O_l10_kl0.0_b1.0_hlnt_a0.0_{}_ep1_e1_he100_fs50_scc_al{}_bl0.0_{}tb5_s1"
SMALL_HTF = Setting(
    key="small_htf", title="Small hard-to-find modes", kind="multiprompt", split="fixed", y_metric="kl_q_sigma",
    runs=(
        _run("ctl", "CTL", _SMALL_HTF.format("-5.0", "ppq_ctl", "3e-06", "ppq_")),
        _run("ctl_temp", "CTL", _SMALL_HTF.format("5.0to-5.0linear", "ppq_ctl", "3e-06", "ppq_"), Tempering("eta", 5, -5)),
        _run("ctl_expl", "CTL", _SMALL_HTF.format("-5.0", "ppq_ctl", "3e-06", "ppq_cf3.0_cd64_cfr0.0003_cfsn_af_fo_"), Exploration(3)),
        _run("rloo", "RLOO", _SMALL_HTF.format("-5.0", "p_reinf", "3e-06", "p_")),
        _run("rloo_temp", "RLOO", _SMALL_HTF.format("5.0to-5.0linear", "p_reinf", "3e-06", "p_"), Tempering("eta", 5, -5)),
        _run("rloo_expl", "RLOO", _SMALL_HTF.format("-5.0", "p_reinf", "3e-06", "p_cf70.0to0.0linear_cd64_cfr0.001_cfsn_af_fo_"), Exploration(70, 0)),
        # appendix-only runs (note: the mixture run used learning rate 1e-6)
        _run("ctl_ent", "CTL", _SMALL_HTF.format("-5.0", "ppq_ctl", "3e-06", "ppq_entb0.003_"), Entropy(0.003), in_logz_pool=False),
        _run("ctl_mix", "CTL", _SMALL_HTF.format("-5.0", "ppq_ctl", "1e-06", "ppq_mixqi_lag10_"), Mixture(10), in_logz_pool=False),
    ),
)

_MEDIUM_HTF = "f_q_g_q_iwae_bounds_OpenRLHF_it{}_Ll3.1BIn_SkReV2Ll3.1B_BaALseshen_l100_kl0.0_b1.0_hlnt_a0.0_{}_ep1_e1_he100_scc_al3e-07_bl0.0_{}tb20_s1"
MEDIUM_HTF = Setting(
    key="medium_htf", title="Medium hard-to-find modes", kind="multiprompt", split="heldout", y_metric="kl_q_sigma",
    runs=(
        _run("ctl", "CTL", _MEDIUM_HTF.format("-5.0", "ppq_ctl", "ppq_")),
        _run("ctl_temp", "CTL", _MEDIUM_HTF.format("5.0to-5.0linear", "ppq_ctl", "ppq_"), Tempering("eta", 5, -5)),
        _run("ctl_expl", "CTL", _MEDIUM_HTF.format("-5.0", "ppq_ctl", "ppq_cf10.0_cd64_cfr0.001_cfsn_cfpSm13In_af_fo_"), Exploration(10)),
        _run("rloo", "RLOO", _MEDIUM_HTF.format("-5.0", "p_reinf", "p_")),
        _run("rloo_temp", "RLOO", _MEDIUM_HTF.format("0.0to-5.0linear", "p_reinf", "p_"), Tempering("eta", 0, -5)),
        _run("rloo_expl", "RLOO", _MEDIUM_HTF.format("-5.0", "p_reinf", "p_cf8.0_cd64_cfr0.001_cfsn_cfpSm13In_af_fo_"), Exploration(8)),
        # appendix-only runs
        _run("ctl_temp_expl", "CTL", _MEDIUM_HTF.format("5.0to-5.0linear", "ppq_ctl", "ppq_cf10.0_cd64_cfr0.001_cfsn_cfpSm13In_af_fo_"),
             Exploration(10), Tempering("eta", 5, -5), in_logz_pool=False),
        _run("dpg", "DPG", _MEDIUM_HTF.format("-5.0", "ppq_ctln", "ppq_"), in_logz_pool=False),
        _run("dpg_temp", "DPG", _MEDIUM_HTF.format("0.0to-5.0linear", "ppq_ctln", "ppq_"), Tempering("eta", 0, -5), in_logz_pool=False),
        _run("dpg_expl", "DPG", _MEDIUM_HTF.format("-5.0", "ppq_ctln", "ppq_cf10.0_cd64_cfr0.001_cfsn_cfpSm13In_af_fo_"), Exploration(10), in_logz_pool=False),
        _run("ctl_u", "CTL(U)", _MEDIUM_HTF.format("-5.0", "ppq_ctlu", "ppq_"), in_logz_pool=False),
    ),
)

SETTINGS = {setting.key: setting for setting in (TOY_MC, TOY_HTF, SMALL_MC, MEDIUM_MC, SMALL_HTF, MEDIUM_HTF)}

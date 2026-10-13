"""Tune Correct and Smooth on validation with Optuna.

Huang et al. (2021) search the diffusion strengths and normalizations on validation, keep the
number of steps fixed and apply the selected setting to every seed. Each stage (both steps,
Correct only, Smooth only) is searched under each residual scaling; the search with the best
validation score gives the reported setting.
"""
from dataclasses import dataclass

import optuna

from rbl import cs

STAGES = ("both", "correct", "smooth")
VALIDATION_DECIMALS = 4
REFERENCE_SETTINGS = (
    {"alpha1": 0.87, "alpha2": 0.81, "A1": "AD", "A2": "DAD"},
    {"alpha1": 0.98, "alpha2": 0.65, "A1": "AD", "A2": "DAD"},
    {"alpha1": 0.9791632871592579, "alpha2": 0.7564990804200602, "A1": "DA", "A2": "AD"},
    {"alpha1": 1.0, "alpha2": 0.9, "A1": "DAD", "A2": "DAD"},
)


@dataclass(frozen=True)
class Search:
    """Outcome of one search: its stage, the best setting and its validation score."""

    stage: str
    setting: cs.Setting
    validation: float


def scalings(binary):
    """Return the residual scalings that apply; autoscale assumes residuals in [-1, 1]."""
    return cs.SCALINGS if binary else tuple(s for s in cs.SCALINGS if s != "autoscale")


def search(score, stage, scaling, trials):
    """Return the Search of one stage under one scaling.

    The settings released with the reference implementation are tried first, then a seeded
    tree-structured Parzen estimator proposes the rest.

    Args:
        score: function of a cs.Setting, higher is better.
        stage: one of STAGES.
        scaling: one of cs.SCALINGS.
        trials: number of settings to evaluate.
    """
    def objective(trial):
        return score(_suggest(trial, stage, scaling))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    for setting in REFERENCE_SETTINGS:
        study.enqueue_trial({**setting, "s": 1.0}, skip_if_exists=True)
    study.optimize(objective, n_trials=trials)
    best = study.best_params
    setting = cs.Setting(scaling, best.get("alpha1", 0.0), best.get("alpha2", 0.0),
                         best.get("A1", "DA"), best.get("A2", "DA"), best.get("s", 1.0))
    return Search(stage, setting, study.best_value)


def select(searches):
    """Return the search with the best validation score at VALIDATION_DECIMALS decimals; ties go
    to the earlier scaling of cs.SCALINGS, then to the earlier stage of STAGES."""
    return min(searches, key=lambda found: (-round(found.validation, VALIDATION_DECIMALS),
                                            cs.SCALINGS.index(found.setting.scaling),
                                            STAGES.index(found.stage)))


def _suggest(trial, stage, scaling):
    alpha1 = trial.suggest_float("alpha1", 0.0, 1.0) if stage != "smooth" else 0.0
    alpha2 = trial.suggest_float("alpha2", 0.0, 1.0) if stage != "correct" else 0.0
    norm1 = trial.suggest_categorical("A1", cs.NORMALIZATIONS) if stage != "smooth" else "DA"
    norm2 = trial.suggest_categorical("A2", cs.NORMALIZATIONS) if stage != "correct" else "DA"
    scale = trial.suggest_float("s", 1.0, 50.0, log=True) if scaling == "fdiff" else 1.0
    return cs.Setting(scaling, alpha1, alpha2, norm1, norm2, scale)

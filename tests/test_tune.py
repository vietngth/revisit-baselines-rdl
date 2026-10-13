import pytest

from rbl import cs, tune


def closeness(setting):
    return -(setting.alpha1 - 0.3) ** 2 - (setting.alpha2 - 0.6) ** 2


def record(stage, scaling, trials=12):
    tried = []

    def score(setting):
        tried.append(setting)
        return closeness(setting)

    return tune.search(score, stage, scaling, trials), tried


def test_reference_settings_are_tried_first():
    _, tried = record("both", "none")
    for setting, expected in zip(tried, tune.REFERENCE_SETTINGS):
        assert (setting.alpha1, setting.alpha2, setting.norm1, setting.norm2) == (
            expected["alpha1"], expected["alpha2"], expected["A1"], expected["A2"])


def test_a_stage_searches_only_its_own_step():
    _, tried = record("correct", "none")
    assert all(setting.alpha2 == 0.0 and setting.norm2 == "DA" for setting in tried)
    assert len({setting.alpha1 for setting in tried}) == len(tried)
    _, tried = record("smooth", "none")
    assert all(setting.alpha1 == 0.0 and setting.norm1 == "DA" for setting in tried)


def test_the_scale_is_searched_only_for_fdiff():
    _, tried = record("both", "fdiff")
    assert tried[0].scale == 1.0 and all(1.0 <= setting.scale <= 50.0 for setting in tried)
    assert len({setting.scale for setting in tried}) > 1
    _, tried = record("both", "autoscale")
    assert {setting.scale for setting in tried} == {1.0}


def test_search_returns_the_best_setting_it_tried():
    found, tried = record("both", "none", trials=30)
    best = max(tried, key=closeness)
    assert found.setting == best and found.stage == "both"
    assert found.validation == pytest.approx(closeness(best))


def test_search_is_reproducible():
    assert record("both", "fdiff", trials=25)[1] == record("both", "fdiff", trials=25)[1]


def test_regression_has_no_autoscale():
    assert tune.scalings(True) == ("none", "fdiff", "autoscale")
    assert tune.scalings(False) == ("none", "fdiff")


def test_select_prefers_the_best_validation_score():
    searches = [tune.Search("both", cs.Setting("none"), 0.70),
                tune.Search("smooth", cs.Setting("autoscale"), 0.71)]
    assert tune.select(searches) is searches[1]


def test_select_breaks_ties_by_scaling_then_stage():
    searches = [tune.Search("smooth", cs.Setting("autoscale"), 0.70004),
                tune.Search("correct", cs.Setting("fdiff"), 0.70001),
                tune.Search("smooth", cs.Setting("none"), 0.69996),
                tune.Search("correct", cs.Setting("none"), 0.69999)]
    assert tune.select(searches) is searches[3]

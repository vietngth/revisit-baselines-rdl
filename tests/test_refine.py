import numpy as np

from rbl import cs, evaluation, label_graphs, labels, projection, refine, tune
from tests.toy import Task, toy_database

SEEDS = 3


def setup(regression=False, tuning=(0, 1, 2)):
    task = Task(regression)
    rows = labels.split_rows(task)
    total = sum(len(rows[split].positions) for split in labels.SPLITS)
    base = np.random.default_rng(3).random((total, SEEDS), dtype=np.float32)
    links = projection.build(toy_database(), "users")
    graphs = {split: label_graphs.build("cross+self", rows[split],
                                        labels.label_history(task, rows, history), links)
              for split, history in (("val", ("train",)), ("test", ("train", "val")))}
    return refine.Refiner(task, base, list(tuning), threads=1), graphs, base, rows


def test_metric_is_auroc_for_classification_and_mae_for_regression():
    classification, regression = evaluation.Metric(Task()), evaluation.Metric(Task(True))
    assert (classification.name, regression.name) == ("roc_auc", "mae")
    labels_test = Task().get_table("test").df["label"].to_numpy()
    assert classification.score(labels_test) == 1.0
    assert regression.score(Task(True).get_table("test").df["label"].to_numpy() + 2.0) == -2.0


def test_metric_falls_back_to_normalized_mae():
    task = Task(True)

    def nmae(true, pred):
        return float(np.mean(np.abs(true - pred))) / 4.0

    task.metrics = [nmae]
    metric = evaluation.Metric(task)
    target = task.get_table("val").df["label"].to_numpy()
    assert metric.name == "nmae" and metric.value(target + 2.0, task.get_table("val")) == 0.5


def test_every_stage_is_tuned_under_every_scaling():
    refiner, graphs, _, _ = setup()
    searches = refiner.tune(graphs["val"], trials=6)
    found = [(search.stage, search.setting.scaling) for search in searches]
    assert found == [(stage, scaling) for scaling in cs.SCALINGS for stage in tune.STAGES]
    refiner, graphs, _, _ = setup(regression=True)
    assert len(refiner.tune(graphs["val"], trials=6)) == 6


def test_a_search_reports_the_validation_score_of_its_setting():
    refiner, graphs, base, _ = setup(regression=True)
    task = Task(True)
    train_targets = task.get_table("train").df["label"]
    bounds = (train_targets.min(), train_targets.max())
    target = task.get_table("val").df["label"].to_numpy()
    for found in refiner.tune(graphs["val"], trials=8):
        refined = cs.correct_and_smooth(graphs["val"], base, found.setting, bounds, False,
                                        threads=1)
        expected = -np.mean([np.mean(np.abs(target - refined[:, seed])) for seed in range(SEEDS)])
        np.testing.assert_allclose(found.validation, expected, rtol=1e-6)


def test_only_the_tuning_seeds_are_scored():
    first, graphs, _, _ = setup(regression=True, tuning=(1,))
    second, _, base, _ = setup(regression=True, tuning=(1,))
    base[:, [0, 2]] += 1.0
    second = refine.Refiner(Task(True), base, [1], threads=1)
    assert first.tune(graphs["val"], trials=6) == second.tune(graphs["val"], trials=6)


def test_test_values_have_one_entry_per_seed():
    refiner, graphs, base, rows = setup()
    values = refiner.test(graphs["test"], cs.Setting(alpha2=0.5))
    assert len(values) == SEEDS and all(0.0 <= value <= 1.0 for value in values)
    untouched = refiner.test(graphs["test"], cs.Setting())
    assert untouched == refiner.test_values(base[rows["test"].positions])

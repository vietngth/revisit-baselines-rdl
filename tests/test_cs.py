import numpy as np
import pytest

from rbl import cs, label_graphs, labels, projection
from tests.toy import Task, toy_database

SEEDS = 3
BINARY_BOUNDS = (0.0, 1.0)


def problem(variant, binary=True, **options):
    """Graph of the test rows and base predictions for SEEDS seeds."""
    task = Task()
    rows = labels.split_rows(task)
    history = labels.label_history(task, rows, ("train", "val"))
    graph = label_graphs.build(variant, rows["test"], history,
                               projection.build(toy_database(), "users"), **options)
    total = sum(len(rows[split].positions) for split in labels.SPLITS)
    rng = np.random.default_rng(1)
    base = rng.random((total, SEEDS), dtype=np.float32)
    return graph, base if binary else 10 * base - 3, rows["test"].positions


def reference(block, base, setting, bounds, binary):
    """Correct and Smooth of one seed with dense float64 matrices, as in the reference code."""
    adjacency = block.adjacency.toarray().astype(np.float64)
    degree = adjacency.sum(1)
    with np.errstate(divide="ignore"):
        inverse = np.where(degree == 0, 0, 1 / degree)
        root = np.where(degree == 0, 0, 1 / np.sqrt(degree))
    operators = {"DA": inverse[:, None] * adjacency, "AD": adjacency * inverse[None, :],
                 "DAD": root[:, None] * adjacency * root[None, :]}
    start = base[block.positions].astype(np.float64)
    labelled = ~np.isnan(block.labels)

    def diffuse(operator, initial, alpha, post):
        state = initial.copy()
        for _ in range(cs.STEPS):
            state = post(alpha * operator @ state + (1 - alpha) * initial)
        return state

    corrected = start
    if setting.alpha1:
        residual = np.where(labelled, block.labels - start, 0.0)
        if setting.scaling == "fdiff":
            def pin(state):
                state[labelled] = residual[labelled]
                return state
            corrected = start + setting.scale * diffuse(operators[setting.norm1], residual,
                                                        setting.alpha1, pin)
        else:
            clip = (lambda state: np.clip(state, -1, 1)) if binary else (lambda state: state)
            diffused = diffuse(operators[setting.norm1], residual, setting.alpha1, clip)
            if binary and setting.scaling == "autoscale":
                scale = np.ones(len(start))
                for row in range(len(block.seeds)):
                    nodes = block.owners == row
                    known = nodes & labelled
                    mean = np.abs(residual[known]).sum() / max(known.sum(), 1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        scale[nodes] = mean / np.abs(diffused[nodes])
                scale[scale > 1000] = 1.0
                scale[np.isnan(scale)] = 0.0
                diffused = scale * diffused
            corrected = start + diffused
    state = np.where(labelled, block.labels, corrected)
    smoothed = diffuse(operators[setting.norm2], state, setting.alpha2,
                       lambda value: np.clip(value, *bounds))
    return np.where(degree[block.seeds] < 1e-12, start[block.seeds], smoothed[block.seeds])


SETTINGS = [cs.Setting("none", 0.6, 0.4, "DA", "AD"),
            cs.Setting("none", 0.9, 0.0, "DAD", "DA"),
            cs.Setting("none", 0.0, 0.7, "DA", "DAD"),
            cs.Setting("autoscale", 0.8, 0.5, "AD", "DA"),
            cs.Setting("autoscale", 0.3, 0.0, "DA", "DA"),
            cs.Setting("fdiff", 0.7, 0.2, "DAD", "AD", 3.0),
            cs.Setting("fdiff", 1.0, 0.9, "DA", "DAD", 20.0)]


@pytest.mark.parametrize("variant", list(label_graphs.VARIANTS))
@pytest.mark.parametrize("binary", [True, False])
def test_matches_the_dense_reference(variant, binary):
    graph, base, _ = problem(variant, binary)
    bounds = BINARY_BOUNDS if binary else (-2.0, 6.0)
    for setting in SETTINGS:
        refined = cs.correct_and_smooth(graph, base, setting, bounds, binary, threads=2)
        expected = np.stack([reference(graph.blocks[0], base[:, seed], setting, bounds, binary)
                             for seed in range(SEEDS)], axis=1)
        np.testing.assert_allclose(refined, expected, rtol=2e-4, atol=2e-5)


def test_zero_strength_returns_the_base():
    for variant in label_graphs.VARIANTS:
        graph, base, positions = problem(variant)
        for scaling in cs.SCALINGS:
            refined = cs.correct_and_smooth(graph, base, cs.Setting(scaling), BINARY_BOUNDS, True)
            np.testing.assert_array_equal(refined, base[positions])


def test_predictions_are_clipped_to_the_bounds():
    graph, base, positions = problem("self", binary=False)
    refined = cs.correct_and_smooth(graph, base, cs.Setting(), (0.0, 1.0), False)
    np.testing.assert_array_equal(refined, np.clip(base[positions], 0.0, 1.0))


def test_full_strength_correct_on_a_star_returns_the_base():
    graph, base, positions = problem("self", binary=False)
    setting = cs.Setting("none", alpha1=1.0, norm1="DAD")
    refined = cs.correct_and_smooth(graph, base, setting, (-100.0, 100.0), False)
    np.testing.assert_array_equal(refined, base[positions])


def test_smoothing_moves_a_seed_towards_its_labels():
    graph, base, positions = problem("self")
    block = graph.blocks[0]
    block.labels[block.labelled] = 1.0
    refined = cs.correct_and_smooth(graph, base, cs.Setting(alpha2=0.5), BINARY_BOUNDS, True)
    assert (refined > base[positions]).all()


def test_regression_residuals_are_not_clipped():
    graph, base, _ = problem("self", binary=False)
    block = graph.blocks[0]
    block.labels[block.labelled] = 40.0
    setting = cs.Setting("none", alpha1=0.5)
    regression = cs.correct_and_smooth(graph, base, setting, (-100.0, 100.0), False)
    binary = cs.correct_and_smooth(graph, base, setting, (-100.0, 100.0), True)
    assert (regression > 10).all() and (binary < 10).all()


def test_seeds_without_labels_keep_their_base():
    task = Task()
    rows = labels.split_rows(task)
    history = labels.label_history(task, rows, ("train",))
    graph = label_graphs.build("self", rows["train"], history)
    base = np.random.default_rng(0).random((len(rows["train"].positions) + 8, 1), dtype=np.float32)
    refined = cs.correct_and_smooth(graph, base, cs.Setting("autoscale", 0.7, 0.7), BINARY_BOUNDS,
                                    True)
    first_day = rows["train"].times == rows["train"].times.min()
    np.testing.assert_array_equal(refined[first_day], base[rows["train"].positions][first_day])
    assert (refined[~first_day] != base[rows["train"].positions][~first_day]).any()


def test_seeds_blocks_and_threads_do_not_change_a_prediction():
    for variant in ("cross+self", "all", "chain+hist"):
        graph, base, _ = problem(variant)
        split, _, _ = problem(variant, block_nodes=1)
        assert len(split.blocks) > len(graph.blocks)
        for setting in SETTINGS:
            together = cs.correct_and_smooth(graph, base, setting, BINARY_BOUNDS, True, threads=1)
            blocks = cs.correct_and_smooth(split, base, setting, BINARY_BOUNDS, True, threads=4)
            np.testing.assert_array_equal(blocks, together)
            for seed in range(SEEDS):
                alone = cs.correct_and_smooth(graph, np.ascontiguousarray(base[:, [seed]]), setting,
                                              BINARY_BOUNDS, True)
                np.testing.assert_array_equal(alone[:, 0], together[:, seed])


def test_future_labels_do_not_reach_a_prediction():
    task = Task()
    rows = labels.split_rows(task)
    graphs = []
    for flipped in (False, True):
        if flipped:
            frame = task.get_table("train").df
            late = frame["time"] >= frame["time"].max()
            frame.loc[late, "label"] = 1 - frame.loc[late, "label"]
        history = labels.label_history(task, rows, ("train",))
        graphs.append(label_graphs.build("chain+hist", rows["val"], history,
                                         projection.build(toy_database(), "users")))
    base = np.random.default_rng(2).random((16, 1), dtype=np.float32)
    setting = cs.Setting("autoscale", 0.8, 0.6, "DAD", "DA")
    clean, changed = (cs.correct_and_smooth(graph, base, setting, BINARY_BOUNDS, True)
                      for graph in graphs)
    np.testing.assert_array_equal(clean, changed)

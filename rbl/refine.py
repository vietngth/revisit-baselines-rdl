"""Correct and Smooth for one task: tuned on its validation rows, applied to its test rows."""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from relbench.base import TaskType

from rbl import cs, evaluation, tune


class Refiner:
    """Refine the base predictions of a task.

    Args:
        task: RelBench entity task.
        base: float32 base predictions stacked over the train, val and test rows, one column
            per seed.
        tuning: columns of `base` that are scored while tuning.
        threads: threads to use; the available cores by default.
    """

    def __init__(self, task, base, tuning, threads=None):
        self.metric = evaluation.Metric(task)
        self._task, self._base = task, base
        self._threads = threads or cs.available_cores()
        self._tuning = np.ascontiguousarray(base[:, tuning])
        self._binary = task.task_type == TaskType.BINARY_CLASSIFICATION
        targets = task.get_table("train").df[task.target_col].astype(float)
        self._bounds = (0.0, 1.0) if self._binary else (float(targets.min()), float(targets.max()))

    def tune(self, graph, trials):
        """Return one tune.Search per stage and scaling, scored on the validation rows that
        `graph` was built for."""
        table = self._task.get_table("val")

        def score(setting):
            refined = np.ascontiguousarray(self._apply(graph, self._tuning, setting).T)
            with ThreadPoolExecutor(self._threads) as pool:
                return float(np.mean(list(pool.map(
                    lambda column: self.metric.score(column, table), refined))))

        searches = []
        for scaling in tune.scalings(self._binary):
            for stage in tune.STAGES:
                start = time.perf_counter()
                searches.append(tune.search(score, stage, scaling, trials))
                print(f"[search] {stage}/{scaling} validation={abs(searches[-1].validation):.4f} "
                      f"{searches[-1].setting} in {time.perf_counter() - start:.0f}s", flush=True)
        return searches

    def test(self, graph, setting):
        """Return the test metric of every seed under `setting`, on the graph of the test rows."""
        return self.test_values(self._apply(graph, self._base, setting))

    def test_values(self, predictions):
        """Return the test metric of test predictions with one column per seed."""
        return [self.metric.value(column) for column in np.ascontiguousarray(predictions.T)]

    def _apply(self, graph, base, setting):
        return cs.correct_and_smooth(graph, base, setting, self._bounds, self._binary,
                                     self._threads)

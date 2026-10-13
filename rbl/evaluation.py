"""The metric a task is tuned and reported on, computed by RelBench."""
from relbench.base import TaskType


class Metric:
    """AUROC for classification; MAE for regression, or normalized MAE for the tasks that report
    only that.

    Args:
        task: RelBench entity task.
    """

    def __init__(self, task):
        functions = {function.__name__: function for function in task.metrics}
        self.regression = task.task_type == TaskType.REGRESSION
        self.name = ("mae" if "mae" in functions else "nmae") if self.regression else "roc_auc"
        self._task, self._function = task, functions[self.name]

    def value(self, predictions, table=None):
        """Return the metric of predictions for `table` (the test table by default)."""
        return float(self._task.evaluate(predictions, table, metrics=[self._function])[self.name])

    def score(self, predictions, table=None):
        """Return the metric signed so that higher is better."""
        value = self.value(predictions, table)
        return -value if self.regression else value

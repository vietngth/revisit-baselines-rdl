import numpy as np

from rbl import labels
from tests.toy import Task, day


def test_seconds_counts_from_1970():
    stamps = np.array(["1970-01-01T00:00:10", "1970-01-02"], "datetime64[s]")
    np.testing.assert_array_equal(labels.seconds(stamps), [10.0, 86400.0])


def test_positions_count_through_the_splits():
    rows = labels.split_rows(Task())
    stacked = np.concatenate([rows[split].positions for split in labels.SPLITS])
    np.testing.assert_array_equal(stacked, np.arange(len(stacked)))


def test_a_label_is_visible_only_after_its_window_closed():
    task = Task()
    rows = labels.split_rows(task)
    history = labels.label_history(task, rows, ("train",))
    closing = labels.seconds(np.array([day(14)]))[0]
    assert len(history.before(0, closing)[0]) == 0
    assert len(history.before(0, closing + 1)[0]) == 1
    assert len(history.before(0, np.inf)[0]) == 3


def test_history_returns_labels_oldest_first_with_their_rows():
    task = Task()
    rows = labels.split_rows(task)
    history = labels.label_history(task, rows, ("train", "val"))
    values, positions = history.before(2, np.inf)
    frame = task.get_table("train").df
    expected = frame[frame["user"] == 2]["label"].tolist() + [task.get_table("val").df["label"][2]]
    assert values == expected
    assert positions == sorted(positions)
    stacked = np.concatenate([rows[split].entities for split in labels.SPLITS])
    assert set(stacked[positions]) == {2}


def test_unknown_entities_have_no_history():
    task = Task()
    history = labels.label_history(task, labels.split_rows(task), ("train",))
    assert len(history.before(99, np.inf)[0]) == 0

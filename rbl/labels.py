"""Task rows and the labels a prediction may read."""
from dataclasses import dataclass

import numpy as np

SPLITS = ("train", "val", "test")


def seconds(stamps):
    """Return timestamps as float seconds since 1970."""
    return np.asarray(stamps).astype("datetime64[ns]").astype(np.int64) / 1e9


@dataclass(frozen=True)
class Rows:
    """Entity, seed time (seconds) and position of the rows of one split.

    Positions count through the splits in SPLITS order, so they index predictions stacked the
    same way.
    """

    entities: np.ndarray
    times: np.ndarray
    positions: np.ndarray


def split_rows(task):
    """Return the Rows of every split of a RelBench entity task."""
    rows, first = {}, 0
    for split in SPLITS:
        frame = task.get_table(split).df
        rows[split] = Rows(frame[task.entity_col].to_numpy(np.int64),
                           seconds(frame[task.time_col]),
                           np.arange(first, first + len(frame)))
        first += len(frame)
    return rows


class LabelHistory:
    """Labels per entity, each readable once the window it describes has closed.

    Args:
        entities: entity of every label.
        visible_at: time (seconds) from which a label may be read.
        labels: the label values.
        positions: position of the row every label belongs to.
    """

    def __init__(self, entities, visible_at, labels, positions):
        order = np.lexsort((visible_at, entities))
        self._visible_at = visible_at[order]
        self._labels = labels[order].astype(np.float32).tolist()
        self._positions = positions[order].tolist()
        owners, starts = np.unique(entities[order], return_index=True)
        stops = np.append(starts[1:], len(order))
        self._spans = dict(zip(owners.tolist(), zip(starts.tolist(), stops.tolist())))

    def before(self, entity, time):
        """Return the lists (labels, positions) of `entity` visible strictly before `time`,
        oldest first."""
        start, stop = self._spans.get(entity, (0, 0))
        stop = start + int(self._visible_at[start:stop].searchsorted(time, side="left"))
        return self._labels[start:stop], self._positions[start:stop]


def label_history(task, rows, splits):
    """Return the LabelHistory of `splits`.

    A row with seed time t is labelled by what happens in (t, t + task.timedelta], so its label
    becomes visible at t + task.timedelta.
    """
    window = task.timedelta.total_seconds()
    return LabelHistory(
        np.concatenate([rows[split].entities for split in splits]),
        np.concatenate([rows[split].times for split in splits]) + window,
        np.concatenate([task.get_table(split).df[task.target_col].to_numpy(np.float64)
                        for split in splits]),
        np.concatenate([rows[split].positions for split in splits]))

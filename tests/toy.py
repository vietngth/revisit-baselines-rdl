"""Small inputs: a collapsed graph, a relational database and an entity task on it."""
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from relbench.base import TaskType
from relbench.metrics import roc_auc

from rbl.graph import CollapsedGraph
from rbl.snapshots import TaskRows

DAY = np.timedelta64(1, "D")
START = np.datetime64("2020-01-01")


def toy_graph(seed=0):
    """Users (static), orders (stamped) and items (static)."""
    rng = np.random.default_rng(seed)
    users, orders, items = 6, 12, 4
    features = {"users": rng.normal(size=(users, 3)).astype(np.float32),
                "orders": rng.normal(size=(orders, 2)).astype(np.float32),
                "items": rng.normal(size=(items, 4)).astype(np.float32)}
    times = {"users": None, "orders": np.arange(orders, dtype=np.int64) * 10, "items": None}
    user_of = rng.integers(0, users, orders)
    item_of = rng.integers(0, items, orders)
    order_nodes = users + np.arange(orders)
    src = np.concatenate([order_nodes, user_of, order_nodes, users + orders + item_of])
    dst = np.concatenate([user_of, order_nodes, users + orders + item_of, order_nodes])
    return CollapsedGraph(["users", "orders", "items"], features, times, src, dst)


def toy_rows():
    nodes = {"train": np.array([0, 1, 2, 3, 0, -1]), "val": np.array([4, 5, 1]),
             "test": np.array([2, 3])}
    seed_times = {"train": np.array([30, 30, 30, 60, 60, 60]), "val": np.array([80, 80, 80]),
                  "test": np.array([110, 110])}
    return TaskRows(nodes, seed_times)


@dataclass
class Table:
    df: pd.DataFrame
    fkey_col_to_pkey_table: dict = field(default_factory=dict)
    pkey_col: str = None
    time_col: str = None


@dataclass
class Database:
    table_dict: dict


def day(offset):
    return START + offset * DAY


def toy_database():
    """Users order items; users also follow each other and belong to a city."""
    users = Table(pd.DataFrame({"user": [0, 1, 2, 3], "city": [0, 0, 1, np.nan]}),
                  {"city": "cities"}, "user")
    cities = Table(pd.DataFrame({"city": [0, 1]}), {}, "city")
    items = Table(pd.DataFrame({"item": [0, 1]}), {}, "item")
    orders = Table(pd.DataFrame({"user": [0, 1, 2, 0, 3, 1], "item": [0, 0, 0, 1, 1, 1],
                                 "time": [day(1), day(2), day(9), day(3), day(4), day(5)]}),
                   {"user": "users", "item": "items"}, None, "time")
    follows = Table(pd.DataFrame({"follower": [0, 2], "followed": [3, 2],
                                  "time": [day(6), day(7)]}),
                    {"follower": "users", "followed": "users"}, None, "time")
    return Database({"users": users, "cities": cities, "items": items, "orders": orders,
                     "follows": follows})


def mae(true, pred):
    return float(np.mean(np.abs(true - pred)))


class Task:
    """Labels of four users over the following week: train at days 7, 14 and 21 (user 1 from
    day 14), validation at day 28, test at day 36. Binary labels, or real ones for regression."""

    entity_col, time_col, target_col, entity_table = "user", "time", "label", "users"
    timedelta = timedelta(days=7)

    def __init__(self, regression=False):
        rng = np.random.default_rng(0)
        self.task_type = TaskType.REGRESSION if regression else TaskType.BINARY_CLASSIFICATION
        self.metrics = [mae] if regression else [roc_auc]
        self._tables = {}
        for split, days in (("train", (7, 14, 21)), ("val", (28,)), ("test", (36,))):
            frame = pd.DataFrame({"user": np.tile(np.arange(4), len(days)),
                                  "time": np.repeat([day(d) for d in days], 4)})
            frame["label"] = (frame["user"] + np.repeat(np.arange(len(days)), 4)) % 2.0
            if regression:
                frame["label"] += rng.normal(size=len(frame))
            self._tables[split] = Table(frame.drop(index=[1] if split == "train" else []))

    def get_table(self, split):
        return self._tables[split]

    def evaluate(self, pred, target_table=None, metrics=None):
        target = (target_table or self._tables["test"]).df[self.target_col].to_numpy()
        return {function.__name__: function(target, pred) for function in metrics or self.metrics}

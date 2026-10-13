"""Co-occurrence graph between the rows of a task's entity table.

Two entities are linked when the schema lets them meet: in rows of one table that point to the
same row of another table (cross routes), in a single row that references both (bridges),
under a shared parent of their own rows (siblings), or as the owner of a row and somebody
referencing that row (three-hop routes). A link carries the time it became observable and the
weight 1 / (k - 1) of a meeting of k entities (Newman, 2001).
"""
import itertools
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rbl import labels
from rbl.atomic_routes import get_atomic_routes

HUB_SIZE = 1000
TOP_NEIGHBOURS = 100
EMPTY = (np.zeros(0, np.int64), np.zeros(0))
ARRAYS = ("entities", "indptr", "others", "times", "weights")


@dataclass
class Projection:
    """Timed, weighted links in compressed sparse row form.

    `entities` holds the linked entity ids in ascending order; the links of `entities[i]` are
    the slice `indptr[i]:indptr[i + 1]` of `others` (indices into `entities`), `times`
    (seconds) and `weights`, sorted by other entity and then by time.
    """

    entities: np.ndarray
    indptr: np.ndarray
    others: np.ndarray
    times: np.ndarray
    weights: np.ndarray
    index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {entity: row for row, entity in enumerate(self.entities.tolist())}

    def save(self, folder):
        """Write one .npy file per array into `folder`, unless another process got there first."""
        temporary = Path(f"{folder}.{os.getpid()}.tmp")
        temporary.mkdir(parents=True)
        for name in ARRAYS:
            np.save(temporary / f"{name}.npy", getattr(self, name))
        try:
            os.replace(temporary, folder)
        except OSError:
            shutil.rmtree(temporary)

    def before(self, entity, time, top=TOP_NEIGHBOURS):
        """Return (entities, weights) of the `top` heaviest neighbours of `entity`, counting
        only links observable strictly before `time`."""
        row = self.index.get(entity)
        if row is None:
            return EMPTY
        links = slice(self.indptr[row], self.indptr[row + 1])
        past = self.times[links] < time
        if not past.any():
            return EMPTY
        others, inverse = np.unique(self.others[links][past], return_inverse=True)
        totals = np.bincount(inverse, weights=self.weights[links][past], minlength=len(others))
        if len(others) > top:
            heaviest = np.argsort(-totals)[:top]
            others, totals = others[heaviest], totals[heaviest]
        return self.entities[others], totals


def build(database, entity_table):
    """Return the Projection of `entity_table` over every route of the schema."""
    links = [_group_links(*_columns(database, table, entity_column, pivot_column))
             for table, entity_column, _, pivot_column in cross_routes(database, entity_table)]
    links += [_bridge_links(*_columns(database, *route))
              for route in bridge_routes(database, entity_table)]
    key = database.table_dict[entity_table].pkey_col
    links += [_group_links(*_columns(database, entity_table, key, column))
              for column in sibling_routes(database, entity_table)]
    links += [_three_hop_links(database, route)
              for route in three_hop_routes(database, entity_table)]
    links = [link for link in links if link is not None]
    if not links:
        return Projection(np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int32),
                          np.zeros(0), np.zeros(0, np.float32))
    first, second, times, weights = (np.concatenate(parts) for parts in zip(*links))
    del links
    entities = np.unique(np.concatenate([first, second]))
    first, second = (np.searchsorted(entities, ids).astype(np.int32) for ids in (first, second))
    sources, others = np.concatenate([first, second]), np.concatenate([second, first])
    del first, second
    times, weights = np.tile(times, 2), np.tile(weights, 2)
    order = np.lexsort((times, others, sources))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(sources, minlength=len(entities)))])
    del sources
    others = others[order]
    times = times[order]
    return Projection(entities, indptr, others, times, weights[order])


def load(folder):
    return Projection(*(np.load(Path(folder) / f"{name}.npy") for name in ARRAYS))


def cross_routes(database, entity_table):
    """Return (table, entity column, pivot table, pivot column) of every table that references
    the entity table and another table.

    RelGNN's atomic routes come first, in its order, followed by the routes it leaves out
    because the table also references itself.
    """
    edge_types = [(name, f"f2p_{column}", parent) for name, table in database.table_dict.items()
                  for column, parent in table.fkey_col_to_pkey_table.items()]
    atomic = []
    for kind, fact, *legs in get_atomic_routes(edge_types):
        if kind == "dim-fact-dim":
            query, query_table, pivot_table, pivot, _ = legs
            if query_table == entity_table and pivot_table != entity_table:
                atomic.append((fact, query.removeprefix("f2p_"), pivot_table,
                               pivot.removeprefix("rev_f2p_")))
    schema = sorted((name, entity_column, pivot_table, pivot_column)
                    for name, table in database.table_dict.items()
                    for entity_column, parent in table.fkey_col_to_pkey_table.items()
                    if parent == entity_table
                    for pivot_column, pivot_table in table.fkey_col_to_pkey_table.items()
                    if pivot_column != entity_column and pivot_table != entity_table)
    return atomic + [route for route in schema if route not in atomic]


def bridge_routes(database, entity_table):
    """Return (table, column, column) of every pair of columns that link two entities in one
    row: two references to the entity table, or the entity table's key and a self-reference."""
    routes = []
    for name, table in database.table_dict.items():
        references = [column for column, parent in table.fkey_col_to_pkey_table.items()
                      if parent == entity_table]
        routes += [(name, *pair) for pair in itertools.combinations(references, 2)]
        if name == entity_table and table.pkey_col is not None:
            routes += [(name, table.pkey_col, column) for column in references]
    return routes


def sibling_routes(database, entity_table):
    """Return the entity table's own references to other tables."""
    table = database.table_dict[entity_table]
    if table.pkey_col is None:
        return []
    return [column for column, parent in table.fkey_col_to_pkey_table.items()
            if parent != entity_table]


def three_hop_routes(database, entity_table):
    """Return (room table, owner column, visitor table, room column, visitor column): a row of
    the room table belongs to one entity and is referenced by rows of other entities."""
    routes = []
    for room_table, rooms in database.table_dict.items():
        owners = [column for column, parent in rooms.fkey_col_to_pkey_table.items()
                  if parent == entity_table]
        if rooms.pkey_col is None or not owners:
            continue
        for visitor_table, visitors in database.table_dict.items():
            references = visitors.fkey_col_to_pkey_table
            routes += [(room_table, owner, visitor_table, room, visitor)
                       for room, room_parent in references.items() if room_parent == room_table
                       for visitor, parent in references.items()
                       if parent == entity_table and visitor != room
                       for owner in owners]
    return routes


def _columns(database, table_name, first, second):
    """Return two columns of a table and its row times, over the rows where none is missing."""
    table = database.table_dict[table_name]
    frame = table.df[[first, second] + ([table.time_col] if table.time_col else [])].dropna()
    times = labels.seconds(frame[table.time_col]) if table.time_col else np.zeros(len(frame))
    return frame[first], frame[second], times


def _group_links(entities, keys, times):
    """Link the entities that share a key, one meeting per key."""
    entities = entities.to_numpy(np.int64)
    codes = pd.factorize(keys)[0]
    order = np.argsort(codes, kind="stable")
    codes, entities, times = codes[order], entities[order], times[order]
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts, stops = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(codes)]])
    several = stops - starts > 1
    links = [_meeting_links(entities[start:stop], times[start:stop])
             for start, stop in zip(starts[several], stops[several])]
    links = [link for link in links if link is not None]
    return tuple(np.concatenate(parts) for parts in zip(*links)) if links else None


def _meeting_links(entities, times):
    """Link every pair of distinct entities of a meeting once, observable from the time both
    have joined it; meetings of more than HUB_SIZE entities are left out."""
    order = np.lexsort((times, entities))
    entities, times = entities[order], times[order]
    joins = np.concatenate([[True], entities[1:] != entities[:-1]])
    entities, times = entities[joins], times[joins]
    size = len(entities)
    if size > HUB_SIZE or size < 2:
        return None
    first, second = np.triu_indices(size, k=1)
    return (entities[first], entities[second], np.maximum(times[first], times[second]),
            np.full(len(first), 1.0 / (size - 1.0), np.float32))


def _bridge_links(first, second, times):
    first, second = first.to_numpy(np.int64), second.to_numpy(np.int64)
    distinct = first != second
    if not distinct.any():
        return None
    return (first[distinct], second[distinct], times[distinct],
            np.ones(distinct.sum(), np.float32))


def _three_hop_links(database, route):
    """Link every visitor of a room to its owner, from the time both rows exist; the room is a
    meeting of its owner and its distinct visitors."""
    room_table, owner, visitor_table, room, visitor = route
    visitors = pd.DataFrame(dict(zip(("room", "visitor", "visit_time"),
                                     _columns(database, visitor_table, room, visitor))))
    key = database.table_dict[room_table].pkey_col
    rooms = pd.DataFrame(dict(zip(("room", "owner", "room_time"),
                                  _columns(database, room_table, key, owner))))
    visits = visitors.merge(rooms, on="room", how="inner")
    visits = visits.astype({"owner": np.int64, "visitor": np.int64})
    visits = visits[visits["owner"] != visits["visitor"]]
    if visits.empty:
        return None
    visits = (visits.assign(time=np.maximum(visits["room_time"], visits["visit_time"]))
              .sort_values("time").drop_duplicates(["room", "visitor"]))
    size = visits.groupby("room")["visitor"].transform("nunique").to_numpy() + 1
    visits, size = visits[size <= HUB_SIZE], size[size <= HUB_SIZE]
    if visits.empty:
        return None
    return (visits["visitor"].to_numpy(), visits["owner"].to_numpy(), visits["time"].to_numpy(),
            (1.0 / (size - 1.0)).astype(np.float32))

import numpy as np
import pandas as pd

from rbl import labels, projection
from tests.toy import Database, Table, day, toy_database


def seconds(offset):
    return labels.seconds(np.array([day(offset)]))[0]


def links_of(built, entity):
    row = built.index[entity]
    span = slice(built.indptr[row], built.indptr[row + 1])
    return list(zip(built.entities[built.others[span]].tolist(), built.times[span].tolist(),
                    built.weights[span].tolist()))


def test_routes_of_the_toy_schema():
    database = toy_database()
    assert projection.cross_routes(database, "users") == [("orders", "user", "items", "item")]
    assert projection.bridge_routes(database, "users") == [("follows", "follower", "followed")]
    assert projection.sibling_routes(database, "users") == ["city"]
    assert projection.three_hop_routes(database, "users") == []


def test_cross_routes_include_tables_that_reference_themselves():
    posts = Table(pd.DataFrame({"post": [0], "owner": [0], "parent": [0]}),
                  {"owner": "users", "parent": "posts"}, "post")
    votes = Table(pd.DataFrame({"user": [0], "post": [0]}), {"user": "users", "post": "posts"})
    users = Table(pd.DataFrame({"user": [0]}), {}, "user")
    database = Database({"users": users, "posts": posts, "votes": votes})
    assert projection.cross_routes(database, "users") == [("votes", "user", "posts", "post"),
                                                           ("posts", "owner", "posts", "parent")]
    assert projection.three_hop_routes(database, "users") == [
        ("posts", "owner", "posts", "parent", "owner"), ("posts", "owner", "votes", "post", "user")]


def test_links_are_symmetric_and_sorted_by_neighbour_then_time():
    built = projection.build(toy_database(), "users")
    for row, entity in enumerate(built.entities.tolist()):
        span = slice(built.indptr[row], built.indptr[row + 1])
        keys = list(zip(built.others[span].tolist(), built.times[span].tolist()))
        assert keys == sorted(keys)
        for other, time, weight in links_of(built, entity):
            assert (entity, time, weight) in links_of(built, other)


def test_a_meeting_of_k_entities_weighs_one_over_k_minus_one():
    built = projection.build(toy_database(), "users")
    shared_items = [(other, time) for other, time, weight in links_of(built, 0) if weight == 0.5]
    assert shared_items == [(1, seconds(2)), (1, seconds(5)), (2, seconds(9)), (3, seconds(4))]
    shared_city = [(other, time) for other, time, weight in links_of(built, 0) if weight == 1.0]
    assert shared_city[0] == (1, 0.0)


def test_a_pair_is_observable_once_both_rows_exist():
    built = projection.build(toy_database(), "users")
    assert links_of(built, 3) == [(0, seconds(4), 0.5), (0, seconds(6), 1.0), (1, seconds(5), 0.5)]


def test_an_entity_joins_a_meeting_once_at_its_first_row():
    users = Table(pd.DataFrame({"user": [0, 1, 2]}), {}, "user")
    items = Table(pd.DataFrame({"item": [0]}), {}, "item")
    orders = Table(pd.DataFrame({"user": [0, 0, 1, 2, 1], "item": [0, 0, 0, 0, 0],
                                 "time": [day(1), day(2), day(3), day(4), day(5)]}),
                   {"user": "users", "item": "items"}, None, "time")
    built = projection.build(Database({"users": users, "items": items, "orders": orders}),
                             "users")
    assert links_of(built, 0) == [(1, seconds(3), 0.5), (2, seconds(4), 0.5)]
    assert links_of(built, 1) == [(0, seconds(3), 0.5), (2, seconds(4), 0.5)]


def test_rows_without_a_shared_key_and_self_links_are_left_out():
    built = projection.build(toy_database(), "users")
    assert all(other != 2 for other, _, _ in links_of(built, 2))
    assert sorted({other for other, _, _ in links_of(built, 2)}) == [0, 1]


def test_neighbours_count_only_past_links():
    built = projection.build(toy_database(), "users")
    neighbours, weights = built.before(0, seconds(4))
    assert neighbours.tolist() == [1] and weights.tolist() == [1.5]
    neighbours, weights = built.before(0, seconds(4) + 1)
    assert neighbours.tolist() == [1, 3] and weights.tolist() == [1.5, 0.5]
    assert built.before(2, seconds(9))[0].size == 0
    assert built.before(2, seconds(9) + 1)[0].tolist() == [0, 1]
    assert built.before(99, seconds(100))[0].size == 0


def test_only_the_heaviest_neighbours_are_kept():
    built = projection.build(toy_database(), "users")
    neighbours, weights = built.before(0, seconds(100), top=2)
    assert neighbours.tolist() == [1, 3] and weights.tolist() == [2.0, 1.5]


def test_hubs_are_left_out(monkeypatch):
    monkeypatch.setattr(projection, "HUB_SIZE", 2)
    built = projection.build(toy_database(), "users")
    assert all(weight != 0.5 for other, _, weight in links_of(built, 0))


def test_three_hop_route_links_visitors_to_the_owner():
    users = Table(pd.DataFrame({"user": [0, 1, 2]}), {}, "user")
    posts = Table(pd.DataFrame({"post": [0], "owner": [0], "time": [day(5)]}),
                  {"owner": "users"}, "post", "time")
    comments = Table(pd.DataFrame({"post": [0, 0, 0, 0], "user": [1, 2, 1, 0],
                                   "time": [day(1), day(8), day(9), day(9)]}),
                     {"post": "posts", "user": "users"}, None, "time")
    built = projection.build(Database({"users": users, "posts": posts, "comments": comments}),
                             "users")
    three_hop = [(1, seconds(5), 0.5), (2, seconds(8), 0.5)]
    same_post = [(1, seconds(9), 0.5), (2, seconds(9), 0.5)]
    assert links_of(built, 0) == sorted(three_hop + same_post)


def test_a_saved_projection_loads_unchanged(tmp_path):
    built = projection.build(toy_database(), "users")
    built.save(tmp_path / "users")
    built.save(tmp_path / "users")
    loaded = projection.load(tmp_path / "users")
    assert [path.name for path in tmp_path.iterdir()] == ["users"]
    for name in projection.ARRAYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(built, name))
        assert getattr(loaded, name).dtype == getattr(built, name).dtype
    assert loaded.before(0, seconds(100))[1].tolist() == built.before(0, seconds(100))[1].tolist()


def test_a_schema_without_routes_gives_an_empty_projection():
    users = Table(pd.DataFrame({"user": [0, 1]}), {}, "user")
    built = projection.build(Database({"users": users}), "users")
    assert built.before(0, 1.0)[0].size == 0

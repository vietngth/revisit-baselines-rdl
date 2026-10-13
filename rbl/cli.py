"""Command line: precompute filtered features, train the MLP, refine with Correct and Smooth."""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from rbl import cs, database, label_graphs, labels, projection, refine, snapshots, tune
from rbl import train as training

SEEDS = "0,7,13,21,42,77,123,555,1234,2026"


def precompute(args):
    dataset, graph = database.load(args.dataset)
    task = get_task(args.dataset, args.task, download=False)
    rows = snapshots.TaskRows.from_task(task, dataset.get_db(), graph)
    print(f"[graph] nodes={graph.num_nodes:,} dim={graph.feature_dim + 2} snapshots="
          f"{len(rows.taus())} k={args.k} shard={args.shard}/{args.num_shards}", flush=True)
    snapshots.run_shard(graph, rows, _feature_dir(args), args.k, args.shard, args.num_shards)


def train(args):
    task = get_task(args.dataset, args.task, download=False)
    matrices = snapshots.assemble(_feature_dir(args))
    targets = task.get_table("train").df[task.target_col].to_numpy().astype(np.float32)
    scorer = training.Scorer(task, targets)
    tests = []
    for seed in _integers(args.seeds):
        start = time.perf_counter()
        score, lr, net = _best_fit(matrices, targets, scorer, seed)
        seconds = time.perf_counter() - start
        predictions = {split: training.predict(net, matrices[split], scorer)
                       for split in labels.SPLITS}
        path = _prediction_path(args, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **predictions)
        tests.append(scorer.metric.value(predictions["test"]))
        _append(args.results, {
            "dataset": args.dataset, "task": args.task, "k": args.k, "seed": seed, "lr": lr,
            "parameters": sum(parameter.numel() for parameter in net.parameters()),
            "train_seconds": round(seconds, 1), "metric": scorer.metric.name,
            "validation": abs(score), "test": tests[-1]})
        print(f"[seed] seed={seed} lr={lr} test={task.evaluate(predictions['test'])}", flush=True)
    print(f"RESULT dataset={args.dataset} task={args.task} k={args.k} "
          f"{scorer.metric.name}={_summary(tests)}", flush=True)


def _best_fit(matrices, targets, scorer, seed):
    """Fit one MLP per learning rate; return (validation score, learning rate, network) of the
    best one."""
    config = training.TrainConfig()
    fits = []
    for lr in config.learning_rates:
        net, score = training.fit(matrices["train"], targets, matrices["val"], scorer, seed, lr,
                                  config)
        print(f"[fit] seed={seed} lr={lr} validation={abs(score):.4f}", flush=True)
        fits.append((score, lr, net))
    return max(fits, key=lambda fitted: fitted[0])


def correct_and_smooth(args):
    task = get_task(args.dataset, args.task, download=False)
    seeds = _integers(args.seeds)
    base = np.stack([np.concatenate([np.load(_prediction_path(args, seed))[split]
                                     for split in labels.SPLITS]).astype(np.float32)
                     for seed in seeds], axis=1)
    tuning = [seeds.index(seed) for seed in _integers(args.tune_seeds or args.seeds)]
    refiner = refine.Refiner(task, base, tuning, args.threads)
    rows = labels.split_rows(task)
    links = _links(args, task)
    searches = refiner.tune(_graph(args, rows["val"], labels.label_history(task, rows, ("train",)),
                                   links, len(tuning)), args.trials)
    selected = tune.select(searches)
    graph = _graph(args, rows["test"], labels.label_history(task, rows, ("train", "val")), links,
                   len(seeds))
    base_tests = refiner.test_values(base[rows["test"].positions])
    for found in searches:
        tests = refiner.test(graph, found.setting)
        for seed, base_test, test in zip(seeds, base_tests, tests):
            _append(args.results, {
                "dataset": args.dataset, "task": args.task, "k": args.k, "graph": args.graph,
                "stage": found.stage, "scaling": found.setting.scaling,
                "alpha1": found.setting.alpha1, "alpha2": found.setting.alpha2,
                "norm1": found.setting.norm1, "norm2": found.setting.norm2,
                "scale": found.setting.scale, "validation": abs(found.validation), "seed": seed,
                "metric": refiner.metric.name, "base": base_test, "refined": test,
                "selected": int(found is selected)})
        if found is selected:
            print(f"RESULT dataset={args.dataset} task={args.task} k={args.k} graph={args.graph} "
                  f"selected={found.stage}/{found.setting.scaling} {refiner.metric.name}: "
                  f"base={_summary(base_tests)} refined={_summary(tests)}", flush=True)


def _links(args, task):
    """Return the co-occurrence projection of the task's entity table, built once per table."""
    if not label_graphs.uses_neighbours(args.graph):
        return None
    folder = Path(args.projections) / f"{args.dataset}_{task.entity_table}"
    if folder.exists():
        return projection.load(folder)
    links = projection.build(get_dataset(args.dataset, download=False).get_db(), task.entity_table)
    links.save(folder)
    return links


def _graph(args, rows, history, links, seeds):
    start = time.perf_counter()
    graph = label_graphs.build(args.graph, rows, history, links, cs.block_nodes(seeds))
    print(f"[graph] {args.graph} rows={len(rows.entities):,} nodes={graph.num_nodes:,} "
          f"in {time.perf_counter() - start:.0f}s", flush=True)
    return graph


def _feature_dir(args):
    return Path(args.features) / f"{args.dataset}_{args.task}_K{args.k}"


def _prediction_path(args, seed):
    return Path(args.predictions) / f"{args.dataset}_{args.task}_K{args.k}_s{seed}.npz"


def _integers(text):
    return [int(value) for value in text.split(",")]


def _summary(values):
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.4f}±{spread:.4f} (n={len(values)})"


def _append(path, row):
    is_new = not Path(path).exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name, handler):
        command = commands.add_parser(name)
        command.set_defaults(handler=handler)
        command.add_argument("--dataset", required=True)
        command.add_argument("--task", required=True)
        command.add_argument("--k", type=int, default=2, help="filter depth; 0 = no filtering")
        return command

    command = add_command("precompute", precompute)
    command.add_argument("--features", default="features")
    command.add_argument("--shard", type=int, default=0)
    command.add_argument("--num-shards", type=int, default=1)

    command = add_command("train", train)
    command.add_argument("--features", default="features")
    command.add_argument("--predictions", default="predictions")
    command.add_argument("--seeds", default=SEEDS, help="comma list")
    command.add_argument("--results", default="results.csv")

    command = add_command("cs", correct_and_smooth)
    command.add_argument("--predictions", default="predictions")
    command.add_argument("--seeds", default=SEEDS, help="comma list")
    command.add_argument("--graph", default="cross+self", choices=list(label_graphs.VARIANTS))
    command.add_argument("--projections", default="projections")
    command.add_argument("--trials", type=int, default=300,
                         help="settings tried per stage and scaling")
    command.add_argument("--tune-seeds", help="seeds scored while tuning; all by default")
    command.add_argument("--threads", type=int, help="the available cores by default")
    command.add_argument("--results", default="cs_results.csv")

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

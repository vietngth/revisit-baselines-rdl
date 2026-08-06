import argparse
import os
import time

import numpy as np
import pandas as pd
import torch


def precompute(args):
    from relbench.tasks import get_task
    os.environ.setdefault("SGC_XNORM", "absnorm")   # the paper protocol; see rbl/sgc.py
    from . import pipeline as P
    t0 = time.perf_counter()
    ds, data = P.load_graph(args.dataset)
    task = get_task(args.dataset, args.task, download=False)
    ntypes = list(data.node_types)
    nnodes = {nt: data[nt].num_nodes for nt in ntypes}
    offs, D, times, n, blocks, dims = P.build_union(data, ntypes, nnodes)
    E, ec, tc = task.entity_table, task.entity_col, task.time_col
    db = ds.get_db()
    et = db.table_dict[E]
    pk = et.df[et.pkey_col].to_numpy()
    pos_of_pk = pd.Series(np.arange(len(pk)), index=pk)
    pos_of_pk = pos_of_pk[~pos_of_pk.index.duplicated()]
    tabs = {s: task.get_table(s).df for s in ("train", "val", "test")}
    used = np.unique(np.concatenate(
        [pos_of_pk.reindex(df[ec].to_numpy()).dropna().to_numpy() for df in tabs.values()]
    )).astype(np.int64)
    pos_local = pd.Series(np.arange(len(used)), index=used)
    ent_pos = offs[E] + used
    stamps = sorted({int(np.datetime64(v, "s").astype("int64"))
                     for s in tabs for v in tabs[s][tc].unique()})
    Dtot = sum(w for (_o, w) in dims.values()) + 2
    rows_of = {s: pos_local.reindex(pos_of_pk.reindex(df[ec].to_numpy()).to_numpy()).to_numpy()
               for s, df in tabs.items()}
    taus_of = {s: df[tc].to_numpy().astype("datetime64[s]").astype("int64")
               for s, df in tabs.items()}
    # TODO: MEMORY BOTTLENECK. 
    Xs = {s: np.zeros((len(df), Dtot), np.float32) for s, df in tabs.items()}
    print(f"[graph] nodes={n:,} D={Dtot} entities_used={len(used):,} "
          f"snapshots={len(stamps)} K={args.k}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    fname = os.path.join(args.out, f"{args.dataset}_{args.task}_K{args.k}.npz")
    sdir = fname[:-4] + ".slices"
    os.makedirs(sdir, exist_ok=True)
    # One SLICE per distinct seed time tau. A slice is the block of rows of X_train/X_val/X_test
    # whose seed time is exactly tau: those rows, and only those, may read from the graph as
    # visible at tau. Slices therefore partition each design matrix and can be computed
    # independently and in any order, which is what makes checkpointing and resumption sound.
    for tau in stamps:
        sf = os.path.join(sdir, f"{tau}.npz")
        if os.path.exists(sf):
            # RESUME PATH. A finished slice stores its row indices (i_*) and their filtered
            # features (x_*); scattering them back reproduces this tau's contribution exactly,
            # so no propagation is repeated. The cost is decompression plus a large scattered
            # write, which is why resuming a wide task is I/O bound rather than cheap.
            z = np.load(sf)
            for s in tabs:
                if f"i_{s}" in z.files:
                    Xs[s][z[f"i_{s}"]] = z[f"x_{s}"]
            print(f"  tau={tau} [checkpoint hit]", flush=True)
            continue
        # COMPUTE PATH. Build the graph visible at tau (rows with time <= tau, degrees
        # recomputed on the masked graph) and propagate: F holds S_tau^K Xbar_tau read out at
        # the task's entity rows. This is the only place where the filter is applied.
        F, nvis, nnz = P.snapshot_feats(data, ntypes, offs, times, n, ent_pos, tau, args.k,
                                        args.norm, blocks, dims, nnodes)
        saved = {}
        for s in tabs:
            m = taus_of[s] == tau          # the rows of split s that belong to this slice
            if not m.any():
                continue
            # their positions among the entities we filtered
            idx = rows_of[s][m]   
            # an entity absent from the graph stays all-zero         
            ok = ~pd.isna(idx)             
            sub = np.zeros((int(m.sum()), Dtot), np.float32)
            sub[ok] = F[idx[ok].astype(np.int64)]
            # scatter into the split's design matrix
            Xs[s][m] = sub                 
            saved[f"i_{s}"] = np.where(m)[0]
            saved[f"x_{s}"] = sub
        np.savez(sf + ".tmp.npz", **saved)
        os.replace(sf + ".tmp.npz", sf)
        print(f"  tau={tau} visible={nvis:,} edges={nnz:,}", flush=True)
        del F                              # the snapshot is large; free it before the next tau
    np.savez(fname, **{f"X_{s}": v for s, v in Xs.items()})
    print(f"[done] wrote {fname} in {time.perf_counter()-t0:.0f}s", flush=True)


def train(args):
    from relbench.base import TaskType
    from relbench.tasks import get_task
    from .train import fit_mlp
    task = get_task(args.dataset, args.task, download=False)
    is_reg = task.task_type == TaskType.REGRESSION
    z = np.load(os.path.join(args.features,
                             f"{args.dataset}_{args.task}_K{args.k}.npz"))
    Xs = {s: z[f"X_{s}"] for s in ("train", "val", "test")}
    tabs = {s: task.get_table(s).df for s in ("train", "val")}
    val_tbl = task.get_table("val")
    tgt = task.target_col
    ytr = tabs["train"][tgt].to_numpy().astype("float32")
    yva = tabs["val"][tgt].to_numpy().astype("float32")
    clamp = tuple(np.nanpercentile(ytr.astype(float), [2, 98])) if is_reg else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] training on {dev}", flush=True)
    if dev == "cpu":
        print("[device] WARNING: no GPU visible. Training will still complete, but on the "
              "widest datasets a fit takes hours on CPU versus minutes on one GPU; submit "
              "this step to a GPU node if you have one.", flush=True)

    os.environ.setdefault("SGC_MLP", "ref")
    os.environ.setdefault("SGC_LAYERS", "2")
    os.environ.setdefault("SGC_BN", "0")
    os.environ.setdefault("SGC_EPOCHS", str(args.epochs))
    os.environ.setdefault("SGC_BS", "512")
    os.environ.setdefault("SGC_LOSS", args.reg_loss)

    lrs = [float(x) for x in args.lr.split(",")]
    best = None
    for lr in lrs:
        os.environ["SGC_LR"] = str(lr)
        net, score, predict, val_true = fit_mlp(Xs["train"], ytr, Xs["val"], yva, task, val_tbl,
                                                args.seed, dev, is_reg, clamp)
        print(f"[lr {lr}] val={val_true:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, val_true, lr, net, predict)
    _, val_true, lr, net, predict = best
    test_tbl = task.get_table("test")
    pt = predict(net, torch.from_numpy(np.ascontiguousarray(Xs["test"])))
    if is_reg and clamp:
        pt = np.clip(pt, clamp[0], clamp[1])
    mt = task.evaluate(pt)
    pv = predict(net, torch.from_numpy(np.ascontiguousarray(Xs["val"])))
    if is_reg and clamp:
        pv = np.clip(pv, clamp[0], clamp[1])
    os.makedirs("preds", exist_ok=True)
    np.savez(os.path.join("preds", f"{args.dataset}_{args.task}_K{args.k}_s{args.seed}.npz"),
             val=pv, test=pt)
    print(f"RESULT dataset={args.dataset} task={args.task} K={args.k} seed={args.seed} "
          f"lr={lr} val={val_true:.4f} test={mt}", flush=True)
    if args.out_csv:
        hdr = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a") as f:
            if hdr:
                f.write("dataset,task,K,seed,lr,val,test_metrics\n")
            f.write(f"{args.dataset},{args.task},{args.k},{args.seed},{lr},"
                    f"{val_true:.6f},\"{mt}\"\n")


def cs(args):
    """Correct-and-Smooth over the entity's own past labels (star or temporal chain)."""
    import pandas as pd
    from relbench.base import TaskType
    from relbench.tasks import get_task
    from .cs import build_meta_graph, correct_and_smooth, _normalize
    task = get_task(args.dataset, args.task, download=False)
    if task.task_type == TaskType.REGRESSION:
        print("[cs] note: on RelBench regression tasks this stage did not improve the base "
              "in our experiments; results are reported for completeness.")
    z = np.load(os.path.join("preds", f"{args.dataset}_{args.task}_K{args.k}_s{args.seed}.npz"))
    tabs = {s: task.get_table(s).df for s in ("train", "val", "test")}
    ent, tcol, tgt = task.entity_col, task.time_col, task.target_col

    # observable label history per entity: train labels for tuning on val,
    # train+val labels for the test pass; always strictly before the seed time
    def refine(split, zs, hist_df):
        df = tabs[split]
        hist = {e: list(zip(g[tgt], g[tcol]))
                for e, g in hist_df.sort_values(tcol, ascending=False).groupby(ent)}
        out = zs.copy().astype(np.float64)
        for i, (e, t) in enumerate(zip(df[ent].to_numpy(), df[tcol].to_numpy())):
            own = [(y, ty) for y, ty in hist.get(e, ()) if ty < t]
            if not own:
                continue
            vals, labs, edges, seed = build_meta_graph(e, t, own, [], args.variant)
            zn = np.full(len(labs), out[i]); y = np.array(labs, np.float64)
            g = correct_and_smooth(zn, y, edges, len(labs), args.alpha1, args.alpha2)
            out[i] = g[seed]
        return out

    hist_train = tabs["train"][[ent, tcol, tgt]]
    hist_tv = pd.concat([hist_train, tabs["val"][[ent, tcol, tgt]]])
    zt = refine("test", z["test"], hist_tv)
    print(f"CS RESULT dataset={args.dataset} task={args.task} K={args.k} seed={args.seed} "
          f"variant={args.variant} a1={args.alpha1} a2={args.alpha2} "
          f"base={task.evaluate(z['test'])} cs={task.evaluate(zt)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(dataset=(str, None), task=(str, None))
    for name, fn in (("precompute", precompute), ("train", train), ("cs", cs)):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--task", required=True)
        p.add_argument("--k", type=int, default=2)
        p.set_defaults(fn=fn)
        if name == "precompute":
            p.add_argument("--norm", default="sym", choices=["sym", "row"])
            p.add_argument("--out", default="features")
        elif name == "train":
            p.add_argument("--features", default="features")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--epochs", type=int, default=40) 
            p.add_argument("--lr", default="0.01,0.003,0.001",
                           help="comma list; selected on validation")
            p.add_argument("--reg-loss", default="l1", choices=["l1", "mse"])
            p.add_argument("--out-csv", default="results.csv")
        else: 
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--variant", default="self", choices=["self", "self-chain"])
            p.add_argument("--alpha1", type=float, default=0.6)
            p.add_argument("--alpha2", type=float, default=0.6)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

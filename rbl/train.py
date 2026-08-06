import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch


def fit_mlp(Xtr, ytr, Xva, yva, task, val_tbl, seed, dev, is_reg, clamp):
    torch.manual_seed(seed)
    arch = os.environ.get("SGC_MLP", "base")
    hid = int(os.environ.get("SGC_HID", "256"))
    if arch == "ref":
        nl = int(os.environ.get("SGC_LAYERS", "3"))
        p_drop = 0.5
        relu_first = os.environ.get("SGC_RELU_FIRST", "1") == "1"
        use_bn = os.environ.get("SGC_BN", "1") == "1"
        mods, d_in = [], Xtr.shape[1]
        for _ in range(nl - 1):
            block = [torch.nn.Linear(d_in, hid)]
            if use_bn:
                block += ([torch.nn.ReLU(), torch.nn.BatchNorm1d(hid)] if relu_first
                          else [torch.nn.BatchNorm1d(hid), torch.nn.ReLU()])
            else:
                block += [torch.nn.ReLU()]
            block += [torch.nn.Dropout(p_drop)]
            mods += block
            d_in = hid
        mods += [torch.nn.Linear(d_in, 1)]
        net = torch.nn.Sequential(*mods).to(dev)
        lr = float(os.environ.get("SGC_LR", "0.01"))
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], hid), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(), torch.nn.Linear(hid, 1)).to(dev)
        lr = float(os.environ.get("SGC_LR", "1e-3"))
    reg_loss = os.environ.get("SGC_LOSS", "l1")
    lf = (torch.nn.MSELoss() if reg_loss == "mse" else torch.nn.L1Loss()) if is_reg \
        else torch.nn.BCEWithLogitsLoss()
    metric = os.environ.get("SGC_TUNE_METRIC", "mae") if is_reg else "roc_auc"
    higher_better = metric != "mae"
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr))
    yt = torch.from_numpy(np.ascontiguousarray(ytr)).float()
    Xv = torch.from_numpy(np.ascontiguousarray(Xva))

    def predict(net_, Xh, bs=200_000):
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xh), bs):
                xb = Xh[i:i + bs].to(dev, non_blocking=True).float()
                o = net_(xb).squeeze(-1)
                outs.append((o if is_reg else torch.sigmoid(o)).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import (BatchSampler, DataLoader, RandomSampler,
                                  SequentialSampler, TensorDataset)

    class LitMLP(pl.LightningModule):
        def __init__(self, net_):
            super().__init__()
            self.net = net_
            self._val = []

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, _):
            xb, yb = batch
            return lf(self.net(xb.float()).squeeze(-1), yb)

        def validation_step(self, batch, _):
            o = self.net(batch[0].float()).squeeze(-1)
            self._val.append((o if is_reg else torch.sigmoid(o)).detach().float().cpu())

        def on_validation_epoch_end(self):
            pv = torch.cat(self._val).numpy() if self._val else np.zeros(0, np.float32)
            self._val.clear()
            if is_reg and clamp:
                pv = np.clip(pv, clamp[0], clamp[1])
            raw = float(task.evaluate(pv, val_tbl)[metric]) if len(pv) else \
                (-1e9 if higher_better else 1e9)
            self._last_raw = raw
            self.log("val_metric", raw if higher_better else -raw, prog_bar=False, batch_size=1)

        def on_train_epoch_start(self):
            sub = getattr(self, "_train_sampler", None)
            if sub is not None and hasattr(sub, "set_epoch"):
                sub.set_epoch(self.current_epoch)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=lr)

    _bs = int(os.environ.get("SGC_BS", "512"))
    _ndev = int(os.environ.get("SGC_DEVICES", "1"))
    if _bs % _ndev:
        raise SystemExit(f"SGC_BS={_bs} must be divisible by SGC_DEVICES={_ndev}")
    _per = _bs // _ndev
    if _ndev > 1:
        print(f"[ddp] {_ndev} devices x per-rank batch {_per} = effective batch {_bs}", flush=True)

    tr_ds, va_ds = TensorDataset(Xt, yt), TensorDataset(Xv)
    if _ndev > 1:
        from torch.utils.data.distributed import DistributedSampler
        _sub = DistributedSampler(tr_ds, shuffle=True, seed=seed, drop_last=False)
    else:
        _sub = RandomSampler(tr_ds)
    tr_dl = DataLoader(tr_ds, batch_size=None, num_workers=0,
                       sampler=BatchSampler(_sub, batch_size=_per, drop_last=False))
    va_dl = DataLoader(va_ds, batch_size=None, num_workers=0,
                       sampler=BatchSampler(SequentialSampler(va_ds), batch_size=200_000,
                                            drop_last=False))
    ckdir = tempfile.mkdtemp(prefix="relcs_ck_")
    ck = ModelCheckpoint(dirpath=ckdir, monitor="val_metric", mode="max", save_top_k=1,
                         filename="best")
    trainer = pl.Trainer(
        max_epochs=int(os.environ.get("SGC_EPOCHS", "40")),
        accelerator=("gpu" if str(dev).startswith("cuda") else "cpu"), devices=_ndev,
        strategy=("ddp" if _ndev > 1 else "auto"),
        callbacks=[EarlyStopping(monitor="val_metric", mode="max", patience=5, min_delta=0.0), ck],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, use_distributed_sampler=False)
    lit = LitMLP(net)
    lit._train_sampler = _sub
    trainer.fit(lit, tr_dl, va_dl)
    best = float(ck.best_model_score) if ck.best_model_score is not None else -1e9
    val_true = best if higher_better else -best
    if ck.best_model_path and os.path.exists(ck.best_model_path):
        lit.load_state_dict(torch.load(ck.best_model_path, map_location="cpu")["state_dict"])
    net = lit.net.to(dev).eval()
    shutil.rmtree(ckdir, ignore_errors=True)
    return net, best, predict, val_true



"""Train the MLP on precomputed features, with early stopping on validation."""
import tempfile
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from rbl import evaluation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREDICT_ELEMENTS = 100_000_000
CLIP_PERCENTILES = (2, 98)


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.5
    batch_size: int = 512
    max_epochs: int = -1
    patience: int = 5
    learning_rates: tuple = (0.01, 0.003, 0.001)


def build_mlp(in_dim, config):
    """`layers` linear layers; each hidden one is followed by ReLU and dropout (no batch norm)."""
    modules, width = [], in_dim
    for _ in range(config.layers - 1):
        modules += [torch.nn.Linear(width, config.hidden), torch.nn.ReLU(),
                    torch.nn.Dropout(config.dropout)]
        width = config.hidden
    modules.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*modules)


class Scorer:
    """Turn raw outputs into predictions and score them on the validation table.

    Classification outputs pass through a sigmoid. Regression predictions are clipped to the
    CLIP_PERCENTILES of the training targets.

    Args:
        task: RelBench entity task.
        targets: float32 training targets.
    """

    def __init__(self, task, targets):
        self.metric = evaluation.Metric(task)
        self._table = task.get_table("val")
        self.is_regression = self.metric.regression
        self.clip = None
        if self.is_regression:
            self.clip = tuple(np.nanpercentile(targets.astype(float), CLIP_PERCENTILES))

    def predictions(self, outputs):
        return outputs if self.is_regression else torch.sigmoid(outputs)

    def clipped(self, predictions):
        return np.clip(predictions, *self.clip) if self.is_regression else predictions

    def validation_score(self, predictions):
        """Higher is better: AUROC for classification, negative MAE for regression."""
        return self.metric.score(self.clipped(predictions), self._table)


class LitMLP(pl.LightningModule):
    def __init__(self, net, loss, scorer, lr):
        super().__init__()
        self.net, self.loss, self.scorer, self.lr = net, loss, scorer, lr
        self.val_outputs = []

    def training_step(self, batch, _):
        features, target = batch
        return self.loss(self.net(features.float()).squeeze(-1), target)

    def validation_step(self, batch, _):
        outputs = self.net(batch[0].float()).squeeze(-1)
        self.val_outputs.append(self.scorer.predictions(outputs).detach().float().cpu())

    def on_validation_epoch_end(self):
        predictions = torch.cat(self.val_outputs).numpy()
        self.val_outputs.clear()
        self.log("val_score", self.scorer.validation_score(predictions), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def fit(x_train, y_train, x_val, scorer, seed, lr, config):
    """Fit one MLP; return the best-validation network and its validation score."""
    torch.manual_seed(seed)
    net = build_mlp(x_train.shape[1], config).to(DEVICE)
    loss = torch.nn.L1Loss() if scorer.is_regression else torch.nn.BCEWithLogitsLoss()
    train_data = TensorDataset(torch.from_numpy(np.ascontiguousarray(x_train)),
                               torch.from_numpy(np.ascontiguousarray(y_train)).float())
    val_data = TensorDataset(torch.from_numpy(np.ascontiguousarray(x_val)))
    train_loader = DataLoader(train_data, batch_size=None, sampler=BatchSampler(
        RandomSampler(train_data), batch_size=config.batch_size, drop_last=False))
    val_loader = DataLoader(val_data, batch_size=None, sampler=BatchSampler(
        SequentialSampler(val_data), batch_size=predict_rows(x_val), drop_last=False))
    model = LitMLP(net, loss, scorer, lr)
    with tempfile.TemporaryDirectory(prefix="rbl_") as folder:
        checkpoint = ModelCheckpoint(dirpath=folder, monitor="val_score", mode="max",
                                     save_top_k=1, filename="best")
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1,
            callbacks=[EarlyStopping(monitor="val_score", mode="max", patience=config.patience),
                       checkpoint],
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, use_distributed_sampler=False)
        trainer.fit(model, train_loader, val_loader)
        best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    return model.net.to(DEVICE).eval(), float(checkpoint.best_model_score)


def predict_rows(features):
    """Rows per inference batch, so that one batch holds about PREDICT_ELEMENTS values."""
    return max(1, PREDICT_ELEMENTS // max(1, features.shape[1]))


def predict(net, features, scorer):
    rows = predict_rows(features)
    features = torch.from_numpy(np.ascontiguousarray(features))
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), rows):
            batch = features[start:start + rows].to(DEVICE, non_blocking=True).float()
            outputs.append(scorer.predictions(net(batch).squeeze(-1)).cpu().numpy())
    predictions = np.concatenate(outputs) if outputs else np.zeros(0, np.float32)
    return scorer.clipped(predictions)

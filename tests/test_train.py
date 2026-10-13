import numpy as np
import pytest
import torch

from rbl import train
from rbl.train import TrainConfig, build_mlp


def count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class Scorer:
    """Scores sigmoid outputs by their negative squared error against fixed targets."""

    is_regression = False

    def __init__(self, targets):
        self.targets = targets

    def predictions(self, outputs):
        return torch.sigmoid(outputs)

    def clipped(self, predictions):
        return predictions

    def validation_score(self, predictions):
        return -float(np.mean((predictions - self.targets) ** 2))


def test_parameter_counts_match_the_paper():
    assert count(build_mlp(622, TrainConfig())) == 159_745
    assert count(build_mlp(1804, TrainConfig())) == 462_337


def test_fit_returns_the_network_with_the_best_validation_score():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(256, 8)).astype(np.float32)
    targets = (features[:, 0] > 0).astype(np.float32)
    scorer = Scorer(targets[:64])
    config = TrainConfig(hidden=8, max_epochs=4, patience=100)
    net, score = train.fit(features, targets, features[:64], scorer, 0, 0.01, config)
    predictions = train.predict(net, features[:64], scorer)
    assert scorer.validation_score(predictions) == pytest.approx(score, rel=1e-5)
    assert score > scorer.validation_score(np.full(64, 0.5, np.float32))

"""Проверки корректности компонента episode logging и его взаимодействия со стендом."""

import numpy as np
from evaluation.logger import EpisodeLogger

def test_logger(tmp_path):
    logger = EpisodeLogger(tmp_path)
    logger.add_step(np.zeros(29), np.zeros(2), np.zeros(8), np.zeros(128))
    logger.save({"success": True})

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "trajectory.npz").exists()

    data = np.load(tmp_path / "trajectory.npz")
    assert data["observations"].shape == (1, 29)

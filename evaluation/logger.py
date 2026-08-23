"""Сохранение наблюдений, действий и диагностики каждого эпизода."""

from pathlib import Path
import json
import time
import numpy as np

class EpisodeLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self.observations = []
        self.positions = []
        self.actions = []
        self.intentions = []
        self.diagnostics = {}
        self.start_time = time.time()

    def add_step(self, observation, position, action, intention, diagnostics=None):
        self.observations.append(np.asarray(observation))
        self.positions.append(np.asarray(position))
        self.actions.append(np.asarray(action))
        self.intentions.append(np.asarray(intention))
        if diagnostics:
            for key, value in diagnostics.items():
                self.diagnostics.setdefault(key, []).append(np.asarray(value))

    def save(self, summary):
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        arrays = {
            "observations": np.asarray(self.observations),
            "positions": np.asarray(self.positions),
            "actions": np.asarray(self.actions),
            "intentions": np.asarray(self.intentions),
        }

        for key, value in self.diagnostics.items():
            arrays["diagnostic_" + key] = np.asarray(value)

        np.savez_compressed(self.output_dir / "trajectory.npz", **arrays)

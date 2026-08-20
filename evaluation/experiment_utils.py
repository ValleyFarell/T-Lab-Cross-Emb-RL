
from pathlib import Path
import json


def create_run_dir(base_dir):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    ids = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            ids.append(int(p.name))

    run_id = max(ids, default=0) + 1
    run_dir = base_dir / f"{run_id:06d}"
    run_dir.mkdir()

    return run_dir


def save_json(path, data):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

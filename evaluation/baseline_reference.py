from pathlib import Path
import json
from datetime import datetime


def save_baseline_reference(results_dir, summary, metadata=None):
    results_dir = Path(results_dir)
    ref_dir = results_dir / "baseline_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "created_at": datetime.now().isoformat(),
        "method": "FB_pi_switch_baseline",
        "summary": summary,
        "metadata": metadata or {},
    }

    path = ref_dir / "baseline_reference.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path

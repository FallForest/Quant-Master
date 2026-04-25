from __future__ import annotations

import json
import pickle
from pathlib import Path


def save_signal_artifact(artifact_path: str | Path, model, metadata: dict) -> Path:
    target = Path(artifact_path)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_signal_artifact(artifact_path: str | Path) -> tuple[object, dict]:
    target = Path(artifact_path)
    model_path = target / "model.pkl"
    metadata_path = target / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"ML signal artifact is incomplete: {target}. Expected model.pkl and metadata.json."
        )
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata

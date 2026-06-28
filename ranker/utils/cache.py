"""Disk caching utilities for precomputed artifacts."""

import pickle
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def save_pickle(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    log.info("Saved pickle: %s", path)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_numpy(arr: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    log.info("Saved numpy: %s (%s)", path, arr.shape)


def load_numpy(path: str) -> np.ndarray:
    return np.load(path)


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def cache_exists(*paths: str) -> bool:
    return all(Path(p).exists() for p in paths)

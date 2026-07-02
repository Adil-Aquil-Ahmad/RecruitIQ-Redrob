"""TurboVec-style embedding compression helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class TurboVecMatrix:
    vectors: np.ndarray
    scales: np.ndarray

    @property
    def shape(self):
        return self.vectors.shape

    def dequantize(self, indices) -> np.ndarray:
        rows = np.asarray(indices, dtype=np.int64)
        if rows.size == 0:
            return np.zeros((0, self.vectors.shape[1]), dtype=np.float32)

        vectors = self.vectors[rows].astype(np.float32)
        scales = self.scales[rows].astype(np.float32)
        if scales.ndim == 1:
            scales = scales[:, None]
        return vectors * scales


def quantize_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    normalized = embeddings / norms

    scales = np.max(np.abs(normalized), axis=1, keepdims=True)
    scales = np.maximum(scales / 127.0, _EPS)
    vectors = np.clip(np.rint(normalized / scales), -127, 127).astype(np.int8)
    return vectors, scales.astype(np.float16)


def save_turbovec_embeddings(embeddings: np.ndarray, path: str) -> None:
    vectors, scales = quantize_embeddings(embeddings)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=vectors, scales=scales)


def load_turbovec_embeddings(path: str) -> TurboVecMatrix:
    with np.load(path, allow_pickle=False) as data:
        return TurboVecMatrix(vectors=data["vectors"], scales=data["scales"])
"""
OOD detection via embedding-based distance scoring.

Embeds triage descriptions using sentence-transformers, then computes
k-NN distances to the calibration set. OOD examples should have
higher distances, which weighted CP uses to widen prediction sets.

New module — no equivalent in the vet demo.
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


class EmbeddingOODDetector:
    """
    OOD detector using embedding distances.

    1. Embed all calibration examples.
    2. For a new input, embed it and compute k-NN distance to calibration.
    3. High distance = OOD. Low distance = ID.

    The distances are fed to weighted CP to widen prediction sets for OOD.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 5,
    ):
        """
        Args:
            model_name: sentence-transformers model name.
                Default: all-MiniLM-L6-v2 (fast, 384-dim).
                Upgrade: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
                         (better medical OOD separation).
            k: Number of nearest neighbors for distance computation.
        """
        self.model_name = model_name
        self.k = k
        self.model = None  # lazy init
        self.cal_embeddings: Optional[np.ndarray] = None
        self.nn_index: Optional[NearestNeighbors] = None

    def _init_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns array of shape (n, dim)."""
        self._init_model()
        return self.model.encode(texts, show_progress_bar=False)

    def fit_calibration(self, cal_texts: List[str]) -> np.ndarray:
        """
        Embed calibration texts and build the k-NN index.

        Args:
            cal_texts: List of calibration symptom descriptions.

        Returns:
            Calibration embeddings of shape (n_cal, dim).
        """
        self.cal_embeddings = self.embed(cal_texts)
        self.nn_index = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn_index.fit(self.cal_embeddings)
        return self.cal_embeddings

    def compute_distances(self, texts: List[str]) -> np.ndarray:
        """
        Compute mean k-NN distance for each text to the calibration set.

        Args:
            texts: List of symptom descriptions.

        Returns:
            Array of shape (n,) with mean k-NN distances.
        """
        if self.nn_index is None:
            raise RuntimeError("Call fit_calibration() first.")

        embeddings = self.embed(texts)
        distances, _ = self.nn_index.kneighbors(embeddings)
        # Mean distance to k nearest neighbors
        return np.mean(distances, axis=1)

    def compute_distances_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute distances from pre-computed embeddings (avoids re-embedding).
        """
        if self.nn_index is None:
            raise RuntimeError("Call fit_calibration() first.")
        distances, _ = self.nn_index.kneighbors(embeddings)
        return np.mean(distances, axis=1)

    def compute_calibration_distances(self) -> np.ndarray:
        """
        Compute k-NN distances for the calibration points themselves.

        Uses leave-one-out: for each cal point, distance to its k nearest
        neighbors (excluding itself). Needed for weighted CP.
        """
        if self.cal_embeddings is None:
            raise RuntimeError("Call fit_calibration() first.")

        # Fit with k+1 to exclude self
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean")
        nn.fit(self.cal_embeddings)
        distances, _ = nn.kneighbors(self.cal_embeddings)
        # Skip the first column (distance to self = 0)
        return np.mean(distances[:, 1:], axis=1)

    def evaluate_ood_separation(
        self,
        id_texts: List[str],
        ood_texts: List[str],
    ) -> dict:
        """
        Evaluate how well embedding distances separate ID from OOD.

        This is Assumption Test 4 from the pilot. We need AUC >= 0.75.

        Args:
            id_texts: In-distribution symptom descriptions.
            ood_texts: Out-of-distribution symptom descriptions.

        Returns:
            Dict with AUC, mean distances, and viability assessment.
        """
        id_distances = self.compute_distances(id_texts)
        ood_distances = self.compute_distances(ood_texts)

        # Binary labels: 0 = ID, 1 = OOD
        labels = np.concatenate([
            np.zeros(len(id_distances)),
            np.ones(len(ood_distances)),
        ])
        all_distances = np.concatenate([id_distances, ood_distances])

        auc = roc_auc_score(labels, all_distances)

        return {
            "auc": float(auc),
            "mean_id_distance": float(np.mean(id_distances)),
            "mean_ood_distance": float(np.mean(ood_distances)),
            "distance_ratio": float(np.mean(ood_distances) / max(np.mean(id_distances), 1e-8)),
            "n_id": len(id_texts),
            "n_ood": len(ood_texts),
            "viable": auc >= 0.75,
            "marginal": 0.65 <= auc < 0.75,
            "kill": auc < 0.65,
        }

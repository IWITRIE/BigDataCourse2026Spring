import numpy as np


class IncrementalSVD:
    """Memory-bounded incremental SGD reference solution."""

    def __init__(self, num_users, num_items, latent_dim, global_mean):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.global_mean = float(global_mean)
        self.user_matrix = None
        self.item_matrix = None
        self.learning_rate = 0.0005
        self.regularization = 0.04
        self.local_epochs = 1
        self.chunk_size = 5_000

    def load_base_model(self, user_matrix: np.ndarray, item_matrix: np.ndarray):
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix

    def update(self, incremental_batch: np.ndarray):
        if incremental_batch.size == 0:
            return

        ratings = np.asarray(incremental_batch, dtype=np.float32)
        lr_reg = self.learning_rate * self.regularization

        for _ in range(self.local_epochs):
            for start in range(0, ratings.shape[0], self.chunk_size):
                end = start + self.chunk_size
                chunk = ratings[start:end]
                u = chunk[:, 0].astype(np.int32)
                i = chunk[:, 1].astype(np.int32)
                r = chunk[:, 2]

                pu = self.user_matrix[u].copy()
                qi = self.item_matrix[i].copy()
                pred = self.global_mean + np.einsum("ij,ij->i", pu, qi)
                err = np.clip(r - pred, -5.0, 5.0).astype(np.float32)

                p_delta = self.learning_rate * (
                    err[:, None] * qi - self.regularization * pu
                )
                q_delta = self.learning_rate * err[:, None] * pu - lr_reg * qi
                np.add.at(self.user_matrix, u, p_delta)
                np.add.at(self.item_matrix, i, q_delta)

    def predict(self, user_id, item_id) -> float:
        if user_id < 0 or user_id >= self.num_users or item_id < 0 or item_id >= self.num_items:
            return self.global_mean
        score = self.global_mean + float(np.dot(self.user_matrix[user_id], self.item_matrix[item_id]))
        return min(5.0, max(0.5, score))

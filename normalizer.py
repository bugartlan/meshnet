import torch
from torch_geometric.data import Data


class Normalizer:
    def __init__(
        self,
        num_features: int,
        num_categorical: int = 1,
        device: str = "cpu",
        stats: dict = None,
    ):
        self.num_features = num_features
        self.device = device
        self.stats = self.load(stats)

        numeric_features = num_features - num_categorical
        idx_mod = torch.arange(numeric_features) % 6

        self.pos_mask = torch.zeros(num_features, dtype=torch.bool)
        self.force_mask = torch.zeros(num_features, dtype=torch.bool)

        self.pos_mask[:numeric_features] = idx_mod < 3
        self.force_mask[:numeric_features] = ~(idx_mod < 3)

        self.F_MAX = 1.0  # Max force for normalization

    def fit(self, graphs: list[Data]):
        all_pos = torch.cat([g.x[:, :3] for g in graphs], dim=0)
        all_edge = torch.cat([g.edge_attr for g in graphs], dim=0)

        all_y = torch.cat([g.y for g in graphs], dim=0)

        self._set_stats(all_pos, all_y, all_edge)

    def _set_stats(self, all_pos, all_y, all_edge):
        """Helper to dictionary-ize the stats."""
        self.pos_mean = all_pos.mean(dim=0).to(self.device)
        self.pos_std = all_pos.std(dim=0).to(self.device) + 1e-6
        self.edge_mean = all_edge.mean(dim=0).to(self.device)
        self.edge_std = all_edge.std(dim=0).to(self.device) + 1e-6
        self.y_mean = all_y.mean(dim=0).to(self.device)
        self.y_std = all_y.std(dim=0).to(self.device) + 1e-6

        self.stats = {
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "e_mean": self.edge_mean,
            "e_std": self.edge_std,
        }

    def _normalize_pos(self, x: torch.Tensor) -> torch.Tensor:
        n, k = x.shape
        x_reshaped = x.reshape(n, -1, 3)
        x_normalized = (x_reshaped - self.pos_mean) / self.pos_std
        return x_normalized.reshape(n, k)

    def _normalize_force(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.F_MAX

    def normalize(self, graph: Data) -> Data:
        g = graph.clone()

        g.x[:, self.pos_mask] = self._normalize_pos(g.x[:, self.pos_mask])
        g.x[:, self.force_mask] = self._normalize_force(g.x[:, self.force_mask])
        g.edge_attr = (g.edge_attr - self.edge_mean) / self.edge_std
        g.y = (g.y - self.y_mean) / self.y_std
        return g

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean

    def load(self, stats: dict):
        if stats is None:
            return None

        stats = {key: value.to(self.device) for key, value in stats.items()}
        self.pos_mean = stats["pos_mean"]
        self.pos_std = stats["pos_std"]
        self.edge_mean = stats["e_mean"]
        self.edge_std = stats["e_std"]
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]

        return stats


class LogNormalizer(Normalizer):
    def fit(self, graphs: list[Data]):
        all_pos = torch.cat([g.x[:, :3] for g in graphs], dim=0)
        all_edge = torch.cat([g.edge_attr for g in graphs], dim=0)

        all_y_raw = torch.cat([g.y for g in graphs], dim=0)
        all_disp = all_y_raw[:, :3]  # x, y, z displacements
        all_stress = all_y_raw[:, 3:]  # Von Mises stress
        all_stress_log = torch.log1p(all_stress)
        all_y_mixed = torch.cat([all_disp, all_stress_log], dim=1)

        self._set_stats(all_pos, all_y_mixed, all_edge)

    def normalize(self, graph: Data) -> Data:
        g = graph.clone()
        g.y[:, 3] = torch.log1p(g.y[:, 3])

        return super().normalize(g)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        log_y = super().denormalize_y(y)
        log_y[:, 3] = torch.expm1(log_y[:, 3])
        return log_y

import torch
from torch_geometric.data import Data


class Normalizer:
    def __init__(self, num_features: int, device: str = "cpu", stats: dict = None):
        self.num_features = num_features
        self.device = device
        self.mask = self._get_mask(num_features)
        self.stats = (
            {key: value.to(device) for key, value in stats.items()} if stats else None
        )

    def _get_mask(self, num_features: int) -> torch.Tensor:
        """Helper to create the boolean mask."""
        mask = torch.ones(num_features, dtype=torch.bool)
        mask[-1] = False  # Skip boundary indicator

        return mask

    def fit(self, graphs: list[Data]):
        all_x = torch.cat([g.x for g in graphs], dim=0)
        all_e = torch.cat([g.edge_attr for g in graphs], dim=0)

        all_y = torch.cat([g.y for g in graphs], dim=0)

        self._set_stats(all_x, all_y, all_e)

    def _set_stats(self, all_x, all_y, all_e):
        """Helper to dictionary-ize the stats."""
        self.stats = {
            "x_mean": all_x.mean(dim=0).to(self.device),
            "x_std": all_x.std(dim=0).to(self.device) + 1e-7,
            "y_mean": all_y.mean(dim=0).to(self.device),
            "y_std": all_y.std(dim=0).to(self.device) + 1e-7,
            "e_mean": all_e.mean(dim=0).to(self.device),
            "e_std": all_e.std(dim=0).to(self.device) + 1e-7,
        }

    def normalize(self, graph: Data) -> Data:
        g = graph.clone()

        g.x[:, self.mask] = (
            g.x[:, self.mask] - self.stats["x_mean"][self.mask]
        ) / self.stats["x_std"][self.mask]
        g.edge_attr = (g.edge_attr - self.stats["e_mean"]) / self.stats["e_std"]
        g.y = (g.y - self.stats["y_mean"]) / self.stats["y_std"]
        return g

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.stats["y_std"] + self.stats["y_mean"]


class LogNormalizer(Normalizer):
    def fit(self, graphs: list[Data]):
        all_x = torch.cat([g.x for g in graphs], dim=0)
        all_e = torch.cat([g.edge_attr for g in graphs], dim=0)

        all_y_raw = torch.cat([g.y for g in graphs], dim=0)
        all_y_log = torch.log1p(all_y_raw)

        self._set_stats(all_x, all_y_log, all_e)

    def normalize(self, graph: Data) -> Data:
        g = graph.clone()
        g.y[:, 3] = torch.log1p(g.y[:, 3])

        return super().normalize(g)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        log_y = super().denormalize_y(y)
        log_y[:, 3] = torch.expm1(log_y[:, 3])
        return log_y

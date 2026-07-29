import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def _physical_node_mask(batch: Data) -> torch.Tensor:
    """Return a mask for the physical-node prefix of every graph in a batch."""
    counts = torch.as_tensor(
        batch.num_physical_nodes, device=batch.x.device, dtype=torch.long
    ).flatten()
    local_indices = (
        torch.arange(batch.num_nodes, device=batch.x.device) - batch.ptr[batch.batch]
    )
    return local_indices < counts[batch.batch]


class Normalizer:
    def __init__(
        self,
        num_features: int,
        num_categorical: int = 1,
        device: str = "cpu",
        stats: dict | None = None,
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
        loader = DataLoader(
            graphs,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=(self.device != "cpu"),  # Speeds up transfer to GPU
        )

        total_nodes = 0
        total_target_nodes = 0
        total_edges = 0
        pos_sum, pos_sq_sum = 0.0, 0.0
        edge_sum, edge_sq_sum = 0.0, 0.0
        y_sum, y_sq_sum = 0.0, 0.0

        for batch in tqdm(loader, desc="Fitting Normalizer"):
            batch = batch.to(self.device)

            pos = batch.x[:, :3]
            edge = batch.edge_attr
            # Virtual-node targets are artificial zero padding and must not
            # influence the scale used for physical prediction targets.
            y = batch.y[_physical_node_mask(batch)]

            total_nodes += pos.shape[0]
            total_target_nodes += y.shape[0]
            total_edges += edge.shape[0]

            pos_sum += pos.sum(dim=0)
            pos_sq_sum += (pos**2).sum(dim=0)

            edge_sum += edge.sum(dim=0)
            edge_sq_sum += (edge**2).sum(dim=0)

            y_sum += y.sum(dim=0)
            y_sq_sum += (y**2).sum(dim=0)

        self.pos_mean = pos_sum / total_nodes
        self.edge_mean = edge_sum / total_edges
        self.y_mean = y_sum / total_target_nodes

        pos_var = (pos_sq_sum / total_nodes) - (self.pos_mean**2)
        self.pos_std = torch.sqrt(torch.clamp(pos_var, min=0.0)) + 1e-6

        edge_var = (edge_sq_sum / total_edges) - (self.edge_mean**2)
        self.edge_std = torch.sqrt(torch.clamp(edge_var, min=0.0)) + 1e-6

        y_var = (y_sq_sum / total_target_nodes) - (self.y_mean**2)
        self.y_std = torch.sqrt(torch.clamp(y_var, min=0.0)) + 1e-6

        self.stats = {
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "e_mean": self.edge_mean,
            "e_std": self.edge_std,
        }

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

    def to(self, device):
        """Move all normalization statistics to the target device."""
        self.device = device
        for key in self.stats:
            self.stats[key] = self.stats[key].to(device)
        # Update internal attribute references
        self.pos_mean = self.stats["pos_mean"]
        self.pos_std = self.stats["pos_std"]
        self.edge_mean = self.stats["e_mean"]
        self.edge_std = self.stats["e_std"]
        self.y_mean = self.stats["y_mean"]
        self.y_std = self.stats["y_std"]
        self.pos_mask = self.pos_mask.to(device)
        self.force_mask = self.force_mask.to(device)
        return self

    def normalize(self, graph: Data) -> Data:
        g = graph.clone()

        g.x[:, self.pos_mask] = self._normalize_pos(g.x[:, self.pos_mask])
        g.x[:, self.force_mask] = self._normalize_force(g.x[:, self.force_mask])
        g.edge_attr = (g.edge_attr - self.edge_mean) / self.edge_std
        g.y = (g.y - self.y_mean) / self.y_std
        return g

    def normalize_(self, graph: Data) -> Data:
        graph.x[:, self.pos_mask] = self._normalize_pos(graph.x[:, self.pos_mask])
        graph.x[:, self.force_mask] = self._normalize_force(graph.x[:, self.force_mask])
        graph.edge_attr = (graph.edge_attr - self.edge_mean) / self.edge_std
        graph.y = (graph.y - self.y_mean) / self.y_std

    def normalize_batch(self, batch: Data) -> Data:
        """In-place normalization of a batched Data object."""
        batch.x[:, self.pos_mask] = (
            batch.x[:, self.pos_mask] - self.pos_mean
        ) / self.pos_std

        # Vectorized Force Normalization
        batch.x[:, self.force_mask] = batch.x[:, self.force_mask] / self.F_MAX

        # Vectorized Edge and Target Normalization
        batch.edge_attr = (batch.edge_attr - self.edge_mean) / self.edge_std
        batch.y = (batch.y - self.y_mean) / self.y_std
        return batch

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        y_std = self.y_std.to(device=y.device, dtype=y.dtype)
        y_mean = self.y_mean.to(device=y.device, dtype=y.dtype)
        return y * y_std + y_mean

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
    @staticmethod
    def _log_stress(stress: torch.Tensor) -> torch.Tensor:
        """Compress signed stress values without producing NaNs."""
        return torch.sign(stress) * torch.log1p(torch.abs(stress))

    @staticmethod
    def _exp_stress(log_stress: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`_log_stress`."""
        return torch.sign(log_stress) * torch.expm1(torch.abs(log_stress))

    def fit(self, graphs: list[Data]):
        all_pos = torch.cat([g.x[:, :3] for g in graphs], dim=0)
        all_edge = torch.cat([g.edge_attr for g in graphs], dim=0)

        all_y_raw = torch.cat([g.y[: int(g.num_physical_nodes)] for g in graphs], dim=0)
        all_disp = all_y_raw[:, :3]  # x, y, z displacements
        all_stress = all_y_raw[:, 3:]
        all_stress_log = self._log_stress(all_stress)
        all_y_mixed = torch.cat([all_disp, all_stress_log], dim=1)

        self._set_stats(all_pos, all_y_mixed, all_edge)

    def normalize(self, graph: Data) -> Data:
        g = graph.clone()
        g.y[:, 3:] = self._log_stress(g.y[:, 3:])
        return super().normalize(g)

    def normalize_(self, graph: Data) -> Data:
        graph.y[:, 3:] = self._log_stress(graph.y[:, 3:])
        super().normalize_(graph)

    def normalize_batch(self, batch: Data) -> Data:
        batch.y[:, 3:] = self._log_stress(batch.y[:, 3:])
        return super().normalize_batch(batch)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        log_y = super().denormalize_y(y)
        log_y[:, 3:] = self._exp_stress(log_y[:, 3:])
        return log_y


def get_physical_node_mask(batch: Data) -> torch.Tensor:
    """Returns a boolean mask selecting physical nodes (excluding virtual ones)."""
    if hasattr(batch, "is_virtual"):
        return batch.is_virtual == 0
    elif hasattr(batch, "num_physical_nodes"):
        counts = torch.as_tensor(
            batch.num_physical_nodes, device=batch.x.device, dtype=torch.long
        ).flatten()
        local_indices = (
            torch.arange(batch.num_nodes, device=batch.x.device)
            - batch.ptr[batch.batch]
        )
        return local_indices < counts[batch.batch]
    return torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.x.device)


class GraphNormalizer(nn.Module):
    """Normalizes Graph Neural Network inputs and outputs for 3D mechanics problems.

    Node Features: [x, y, z, fx, fy, fz, is_boundary, is_virtual]
    Edge Features: [dx, dy, dz, distance]
    Output Features: [dx', dy', dz', sx, sy, sz, txy, txz, tyz]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Persistent normalization statistics
        self.register_buffer("pos_mean", torch.zeros(3))
        self.register_buffer("pos_std", torch.ones(3))

        self.register_buffer("force_max", torch.tensor(1.0))

        self.register_buffer("edge_mean", torch.zeros(4))
        self.register_buffer("edge_std", torch.ones(4))

        self.register_buffer("disp_mean", torch.zeros(3))
        self.register_buffer("disp_std", torch.ones(3))

        self.register_buffer("stress_log_mean", torch.zeros(6))
        self.register_buffer("stress_log_std", torch.ones(6))

        self.fitted = False

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        """Symmetric log transform: preserves sign, compresses large magnitudes."""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def _symexp(x: torch.Tensor) -> torch.Tensor:
        """Inverse symmetric log transform."""
        return torch.sign(x) * torch.expm1(torch.abs(x))

    @torch.no_grad()
    def fit(self, dataset: list[Data], batch_size: int = 64, num_workers: int = 4):
        """Computes means and standard deviations across the dataset using Welford's algorithm."""
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Running accumulators for Welford's algorithm
        pos_count, pos_m, pos_m2 = 0, torch.zeros(3), torch.zeros(3)
        edge_count, edge_m, edge_m2 = 0, torch.zeros(4), torch.zeros(4)
        disp_count, disp_m, disp_m2 = 0, torch.zeros(3), torch.zeros(3)
        stress_count, stress_m, stress_m2 = 0, torch.zeros(6), torch.zeros(6)

        max_force = 0.0

        for batch in tqdm(loader, desc="Fitting Normalizer"):
            # 1. Spatial Positions [x, y, z]
            pos = batch.x[:, :3].cpu()
            pos_count, pos_m, pos_m2 = self._welford_update(
                pos, pos_count, pos_m, pos_m2
            )

            # 2. Maximum Applied Force Magnitude
            forces = batch.x[:, 3:6].abs().max().item()
            max_force = max(max_force, forces)

            # 3. Edge Features [dx, dy, dz, distance]
            edges = batch.edge_attr.cpu()
            edge_count, edge_m, edge_m2 = self._welford_update(
                edges, edge_count, edge_m, edge_m2
            )

            # 4. Target Features (Physical nodes only)
            if hasattr(batch, "y") and batch.y is not None:
                mask = get_physical_node_mask(batch).cpu()
                y_phys = batch.y[mask].cpu()

                disp = y_phys[:, :3]
                stress_log = self._symlog(y_phys[:, 3:9])

                disp_count, disp_m, disp_m2 = self._welford_update(
                    disp, disp_count, disp_m, disp_m2
                )
                stress_count, stress_m, stress_m2 = self._welford_update(
                    stress_log, stress_count, stress_m, stress_m2
                )

        # Store calculated statistics
        self.pos_mean.copy_(pos_m)
        self.pos_std.copy_(
            torch.sqrt(pos_m2 / max(pos_count - 1, 1)).clamp_min(self.eps)
        )

        self.force_max.copy_(torch.tensor(max_force if max_force > 0 else 1.0))

        self.edge_mean.copy_(edge_m)
        self.edge_std.copy_(
            torch.sqrt(edge_m2 / max(edge_count - 1, 1)).clamp_min(self.eps)
        )

        if disp_count > 0:
            self.disp_mean.copy_(disp_m)
            self.disp_std.copy_(
                torch.sqrt(disp_m2 / max(disp_count - 1, 1)).clamp_min(self.eps)
            )

            self.stress_log_mean.copy_(stress_m)
            self.stress_log_std.copy_(
                torch.sqrt(stress_m2 / max(stress_count - 1, 1)).clamp_min(self.eps)
            )

        self.fitted = True

    @staticmethod
    def _welford_update(
        x: torch.Tensor, count: int, mean: torch.Tensor, M2: torch.Tensor
    ):
        """Numerically stable online update for mean and variance (Welford's algorithm)."""
        n = x.size(0)
        if n == 0:
            return count, mean, M2

        new_count = count + n
        x_mean = x.mean(dim=0)
        delta = x_mean - mean
        new_mean = mean + delta * (n / new_count)

        # Sum of squared differences from the mean
        M2_x = ((x - x_mean) ** 2).sum(dim=0)
        new_M2 = M2 + M2_x + (delta**2) * (count * n / new_count)

        return new_count, new_mean, new_M2

    def normalize(self, data: Data) -> Data:
        """Normalizes a PyG Data or Batch object (creates a clone)."""
        out = data.clone()

        # 1. Node features: [x, y, z] -> Z-score
        out.x[:, :3] = (out.x[:, :3] - self.pos_mean) / self.pos_std

        # 2. Node features: [fx, fy, fz] -> Max scaling
        out.x[:, 3:6] = out.x[:, 3:6] / self.force_max

        # 3. Node flags [is_boundary, is_virtual] remain untouched (0 or 1)

        # 4. Edge features -> Z-score
        out.edge_attr = (out.edge_attr - self.edge_mean) / self.edge_std

        # 5. Targets [dx, dy, dz, stress...]
        if hasattr(out, "y") and out.y is not None:
            disp_norm = (out.y[:, :3] - self.disp_mean) / self.disp_std
            stress_log = self._symlog(out.y[:, 3:9])
            stress_norm = (stress_log - self.stress_log_mean) / self.stress_log_std

            out.y = torch.cat([disp_norm, stress_norm], dim=-1)

        return out

    def denormalize_y(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Converts model predictions back to physical units (meters and Pascals)."""
        disp_pred = y_pred[:, :3]
        stress_pred = y_pred[:, 3:9]

        # Invert Z-score scaling
        disp_physical = (disp_pred * self.disp_std) + self.disp_mean
        stress_log = (stress_pred * self.stress_log_std) + self.stress_log_mean

        # Invert symmetric log
        stress_physical = self._symexp(stress_log)

        return torch.cat([disp_physical, stress_physical], dim=-1)

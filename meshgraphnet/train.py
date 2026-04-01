import argparse
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nets import EncodeProcessDecode, MeshGraphNet
from normalizer import LogNormalizer, Normalizer
from utils import get_weight

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


LATENT_DIM = 128
USE_LAYER_NORM = True


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architectural hyperparameters for the EncodeProcessDecode model."""

    node_dim: int
    edge_dim: int
    output_dim: int
    num_categorical: int
    latent_dim: int = LATENT_DIM
    message_passing_steps: int = 15
    use_layer_norm: bool = USE_LAYER_NORM

    def as_checkpoint_dict(self) -> dict[str, Any]:
        """Serialise fields to a flat dict for checkpoint storage."""
        return {
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "message_passing_steps": self.message_passing_steps,
            "use_layer_norm": self.use_layer_norm,
            "num_categorical": self.num_categorical,
        }


@dataclass
class CheckpointConfig:
    """Everything needed to write an intermediate checkpoint during training."""

    directory: Path
    model_config: ModelConfig
    normalizer: Normalizer
    args: argparse.Namespace


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args():
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(description="Train model on a graph dataset.")

    # --- IO ---
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/"),
        help="Directory to save the trained model file.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Filename for the trained model file.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="Primitives100",
        help="Name of the graph dataset file (without extension) or folder.",
    )

    # --- Loss Configuration ---
    p.add_argument(
        "--log-loss",
        action="store_true",
        help="Use log scaling for loss computation.",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use weighted MSE loss based on distance to the bottom.",
    )
    p.add_argument(
        "--target",
        choices=["all", "displacement", "stress"],
        default="all",
        help="Which output components to include in the loss.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=100.0,
        help="Exponential scaling factor for the weighted loss.",
    )

    # --- Training Hyperparameters ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--layers",
        type=int,
        default=15,
        help="Number of message-passing steps.",
    )

    # --- Runtime Flags ---
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (e.g. 'cpu' or 'cuda').",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/"),
        help="Directory to save TensorBoard logs.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print model parameter shapes on startup.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, PyTorch (CPU and GPU) for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_directories(args: argparse.Namespace, checkpoint_dir: Path) -> None:
    """Create all output directories required before training starts."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard:
        args.log_dir.mkdir(parents=True, exist_ok=True)


def resolve_model_name(args: argparse.Namespace) -> str:
    """Derive a model filename stem from training arguments when no name is given."""
    if args.output_name:
        return args.output_name
    suffix = "w" if args.weighted_loss else "uw"
    return f"{args.dataset}_{args.target}_{suffix}"


# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------


def load_graphs_and_params(dataset_name: str) -> tuple[list[Any], dict[str, Any], Path]:
    """Load graph objects and dataset parameters from disk.

    Supports both a single ``.pt`` file and a directory of ``.pt`` files.

    Args:
        dataset_name: Dataset filename stem or folder name under ``data/``.

    Returns:
        Tuple of (graphs, params dict, resolved dataset path).

    Raises:
        ValueError: If no graphs are found or the directory contains no ``.pt`` files.
    """
    dataset_path = Path("data") / dataset_name

    if dataset_path.is_dir():
        graphs: list[Any] = []
        params: dict[str, Any] | None = None

        for pt_file in sorted(dataset_path.glob("*.pt")):
            loaded = torch.load(pt_file, weights_only=False)
            if params is None:
                params = loaded["params"]
            graphs.extend(loaded["graphs"])

        if params is None:
            raise ValueError(f"# .pt files found in dataset folder: {dataset_path}")
    else:
        dataset_path = Path("data") / f"{dataset_name}.pt"
        loaded = torch.load(dataset_path, weights_only=False)
        graphs = loaded["graphs"]
        params = loaded["params"]

    if not graphs:
        raise ValueError(f"No graphs loaded from dataset: {dataset_path}")

    return graphs, params, dataset_path


def get_target_indices(target: str) -> list[int]:
    """Map a target name to the corresponding output column indices.

    Args:
        target: One of ``"all"``, ``"displacement"``, or ``"stress"``.

    Returns:
        List of integer column indices.

    Raises:
        ValueError: If *target* is not recognised.
    """
    targets: dict[str, list[int]] = {
        "all": [0, 1, 2, 3],
        "displacement": [0, 1, 2],
        "stress": [3],
    }
    if target not in targets:
        raise ValueError(f"Unknown target: {target}")
    return targets[target]


def prepare_graphs(
    graphs, normalizer, weighted_loss: bool, alpha: float, num_targets: int
):
    """Normalise graphs and attach per-node importance weights.

    Args:
        graphs: Raw graph objects.
        normalizer: Fitted normalizer instance.
        weighted_loss: Whether to use distance-based weighting.
        alpha: Exponential scaling factor for the weighted mode.
        num_targets: Number of target output dimensions (sets weight column count).

    Returns:
        List of normalised graph objects with a ``weight`` attribute.
    """
    mode = "weighted" if weighted_loss else "all"

    for graph in tqdm(graphs, desc="Preparing graphs", dynamic_ncols=True):
        weight = get_weight(
            graph.x[:, 2],
            num_targets,
            mode=mode,
            alpha=alpha,
        )
        weight.mul_((graph.x[:, -1] != 1.0).unsqueeze(1).float())
        normalizer.normalize_(graph)
        graph.weight = weight


def prepare_graphs_fast(
    graphs,
    normalizer,
    weighted_loss: bool,
    alpha: float,
    num_targets: int,
    device="cpu",
):
    """Refactored to process all graphs at once on the GPU/CPU."""
    mode = "weighted" if weighted_loss else "all"

    # 1. Collate all individual graphs into one giant Batch
    batch = Batch.from_data_list(graphs).to(device)
    normalizer.to(device)

    # 2. Compute weights for EVERY node in the dataset simultaneously
    # This assumes get_weight is written using torch operations
    weights = get_weight(
        batch.x[:, 2],
        num_targets,
        mode=mode,
        alpha=alpha,
    )

    # 3. Apply physical node mask (vectorized)
    # batch.x[:, -1] is the categorical 'node type' column
    is_physical = (batch.x[:, -1] != 1.0).unsqueeze(1).float()
    batch.weight = weights * is_physical

    # 4. Normalize the entire batch in one call
    batch = normalizer.normalize_batch(batch)

    # 5. Explode back into a list of individual graphs for the DataLoader
    graphs_out = batch.to_data_list()
    for i, g in enumerate(graphs_out):
        s = int(batch.ptr[i].item())
        e = int(batch.ptr[i + 1].item())
        g.weight = weights[s:e]

    return graphs_out


# ---------------------------------------------------------------------------
# Model / optimizer factories
# ---------------------------------------------------------------------------


def build_normalizer(
    use_log_loss: bool,
    num_features: int,
    num_categorical: int,
    device: str = "cpu",
) -> Normalizer:
    """Instantiate the appropriate normalizer based on the loss flag.

    Args:
        use_log_loss: If ``True``, return a ``LogNormalizer``.
        num_features: Total number of node feature dimensions.
        num_categorical: Number of categorical (non-normalised) features.
        device: Compute device for normalizer statistics tensors.

    Returns:
        An unfitted ``Normalizer`` or ``LogNormalizer``.
    """
    cls = LogNormalizer if use_log_loss else Normalizer
    return cls(
        num_features=num_features, num_categorical=num_categorical, device=device
    )


def build_model(config: ModelConfig, device: torch.device) -> EncodeProcessDecode:
    """Construct and move an ``EncodeProcessDecode`` model to *device*.

    Args:
        config: Architectural hyperparameters.
        device: Target compute device.

    Returns:
        Initialised model in training mode.
    """
    return EncodeProcessDecode(
        node_dim=config.node_dim,
        edge_dim=config.edge_dim,
        output_dim=config.output_dim,
        latent_dim=config.latent_dim,
        message_passing_steps=config.message_passing_steps,
        use_layer_norm=config.use_layer_norm,
    ).to(device)


def build_optimizer(
    model: EncodeProcessDecode, learning_rate: float
) -> torch.optim.Adam:
    """Create an Adam optimiser for *model*.

    Args:
        model: Model whose parameters will be optimised.
        learning_rate: Initial learning rate.

    Returns:
        Configured ``Adam`` optimiser.
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ExponentialLR:
    """Create an exponential LR scheduler with a fixed gamma.

    Args:
        optimizer: Optimiser to wrap.

    Returns:
        ``ExponentialLR`` scheduler (gamma = 0.998).
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)


def create_tensorboard_writer(args: argparse.Namespace, model_name: str):
    """Optionally create a TensorBoard ``SummaryWriter``.

    Args:
        args: Parsed CLI arguments.
        model_name: Subdirectory name within the log directory.

    Returns:
        ``(writer, log_path)`` if TensorBoard is enabled, else ``(None, None)``.
    """
    if not args.tensorboard:
        return None, None
    log_path = args.log_dir / model_name
    writer = SummaryWriter(log_dir=log_path)
    print(f"TensorBoard logging to: {log_path}")
    return writer, log_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: EncodeProcessDecode,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target_indices: list[int],
    device: torch.device,
    use_amp: bool,
) -> float:
    """Run one full pass over *loader* and return the average weighted MSE loss.

    Args:
        model: Model in training mode.
        loader: DataLoader yielding batched graphs.
        optimizer: Gradient optimiser.
        scaler: AMP gradient scaler (no-op when ``use_amp=False``).
        target_indices: Output column indices included in the loss.
        device: Compute device.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Average loss per node across all batches.

    Raises:
        ValueError: If a NaN loss is detected.
    """
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with autocast_ctx:
            y_pred = model(batch)[:, target_indices]
            y_true = batch.y[:, target_indices]
            loss = F.mse_loss(y_pred, y_true, weight=batch.weight)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Check data and model for issues.")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

    return total_loss / total_nodes


def train_model(
    model: EncodeProcessDecode,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    target_indices: list[int],
    device: torch.device,
    num_epochs: int,
    writer: SummaryWriter | None,
    checkpoint_cfg: CheckpointConfig,
    checkpoint_interval: int = 50,
) -> list[float]:
    """Train *model* for *num_epochs* epochs and return the per-epoch loss history.

    Args:
        model: Model to train (set to training mode internally).
        loader: DataLoader over normalised training graphs.
        optimizer: Gradient optimiser.
        scheduler: Learning-rate scheduler stepped once per epoch.
        scaler: AMP gradient scaler.
        target_indices: Output column indices included in the loss.
        device: Compute device.
        num_epochs: Total number of training epochs.
        writer: Optional TensorBoard writer; ``None`` disables TB logging.
        checkpoint_cfg: Configuration for periodic checkpoint saving.
        checkpoint_interval: Save a checkpoint every this many epochs.

    Returns:
        List of average per-node losses, one entry per epoch.
    """
    model.train()
    use_amp = device.type == "cuda"
    loss_history: list[float] = []

    progress_bar = tqdm(range(num_epochs), dynamic_ncols=True)

    for epoch in progress_bar:
        avg_loss = train_one_epoch(
            model, loader, optimizer, scaler, target_indices, device, use_amp
        )
        scheduler.step()
        loss_history.append(avg_loss)

        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_cfg.directory / f"model{epoch + 1}.pth"
            save_checkpoint(checkpoint_path, model, checkpoint_cfg)
            print(f"Saved checkpoint at epoch {epoch + 1}: {checkpoint_path}")

        if (epoch + 1) % 10 == 0:
            progress_bar.set_description(
                f"Epoch {epoch + 1:>6}/{num_epochs}, Loss: {avg_loss:.6f}"
            )

    return loss_history


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_path: Path,
    model: EncodeProcessDecode,
    cfg: CheckpointConfig,
) -> None:
    """Serialise model weights, architecture config, and training metadata.

    Args:
        output_path: Destination ``.pth`` file path.
        model: Trained (or partially trained) model.
        cfg: Checkpoint configuration carrying normalizer and training args.
    """
    args = cfg.args
    torch.save(
        {
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "params": cfg.model_config.as_checkpoint_dict(),
            "normalizer": cfg.normalizer.__class__.__name__,
            "stats": cfg.normalizer.stats,
            "training_args": {
                "num_epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "target": args.target,
                "weighted_loss": args.weighted_loss,
                "alpha": args.alpha,
                "log_loss": args.log_loss,
            },
        },
        output_path,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    model_name = resolve_model_name(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    start = time.time()
    graphs, params, dataset_path = load_graphs_and_params(args.dataset)
    print(
        f"Loaded {len(graphs)} graphs from: {dataset_path} in {time.time() - start:.2f}s."
    )

    # Normalizer
    normalizer = build_normalizer(
        args.log_loss, params["node_dim"], params["num_categorical"]
    )
    normalizer.fit(graphs)

    # Data
    target_indices = get_target_indices(args.target)
    start = time.time()
    graphs = prepare_graphs_fast(
        graphs, normalizer, args.weighted_loss, args.alpha, len(target_indices)
    )
    print(f"Normalized {len(graphs)} graphs in {time.time() - start:.2f}s.")

    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )
    print(f"DataLoader created with batch size {args.batch_size}.")

    # Model
    model_config = ModelConfig(
        node_dim=params["node_dim"],
        edge_dim=params["edge_dim"],
        output_dim=params["output_dim"],
        num_categorical=params["num_categorical"],
        message_passing_steps=args.layers,
    )
    model = build_model(model_config, device)

    if args.debug:
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.dtype}, shape={list(param.shape)}")

    # Optimiser / scheduler / AMP
    optimizer = build_optimizer(model, args.learning_rate)
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # Directories + TensorBoard
    checkpoint_dir = args.log_dir / model_name / time.strftime("%Y%m%d-%H%M%S")
    setup_directories(args, checkpoint_dir)
    writer, log_path = create_tensorboard_writer(args, model_name)
    print(f"Checkpoint directory: {checkpoint_dir}")

    checkpoint_cfg = CheckpointConfig(
        directory=checkpoint_dir,
        model_config=model_config,
        normalizer=normalizer,
        args=args,
    )

    # Train
    start = time.time()
    train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        target_indices=target_indices,
        device=device,
        num_epochs=args.epochs,
        writer=writer,
        checkpoint_cfg=checkpoint_cfg,
        checkpoint_interval=50,
    )
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.2f}s.")

    # Final checkpoint
    final_path = args.output_dir / f"{model_name}.pth"
    save_checkpoint(final_path, model, checkpoint_cfg)
    print(f"Model saved to: {final_path}")

    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {log_path}")


if __name__ == "__main__":
    main()

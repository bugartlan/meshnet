import argparse
import random
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from physicsnemo.models.meshgraphnet import (
    BiStrideMeshGraphNet,
    MeshGraphKAN,
    MeshGraphNet,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from normalizer import LogNormalizer, Normalizer
from utils import get_weight

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Train model on a graph dataset.")

    # --- IO Configuration ---
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/"),
        help="Directory to save the trained model file.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Filename for the trained model file.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cantilever_1_10",
        help="Name of the graph dataset file (without extension).",
    )

    # --- Training Configuration ---
    p.add_argument(
        "--log-loss",
        action="store_true",
        help="Whether to use log scaling for the loss computation.",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Whether to use weighted MSE loss based on distance to the bottom.",
    )
    p.add_argument(
        "--target",
        choices=["all", "displacement", "stress"],
        default="all",
        help="Which components to include in the loss calculation.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Exponential scaling factor (only used if --weight-mode='weighted').",
    )

    # --- Training Hyperparameters ---
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--layers",
        type=int,
        default=15,
        help="Number of message passing steps in the model.",
    )

    # --- Runtime Flags ---
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Whether to show the training loss plot.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print debug information.",
    )

    return p.parse_args()


LATENT_DIM = 128
USE_LAYER_NORM = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    return f"{args.dataset}_{args.target}_{'w' if args.weighted_loss else 'uw'}"


def load_graphs_and_params(dataset_name: str) -> tuple[list[Any], dict[str, Any], Path]:
    dataset_path = Path("data") / dataset_name

    if dataset_path.is_dir():
        graphs: list[Any] = []
        params = None

        for pt_file in sorted(dataset_path.glob("*.pt")):
            loaded = torch.load(pt_file, weights_only=False)
            if params is None:
                params = loaded["params"]
            graphs.extend(loaded["graphs"])

        if params is None:
            raise ValueError(f"No .pt files found in dataset folder: {dataset_path}")
    else:
        dataset_path = Path("data") / f"{dataset_name}.pt"
        loaded = torch.load(dataset_path, weights_only=False)
        graphs = loaded["graphs"]
        params = loaded["params"]

    if not graphs:
        raise ValueError(f"No graphs loaded from dataset: {dataset_path}")

    return graphs, params, dataset_path


def build_normalizer(
    use_log_loss: bool, num_features: int, num_categorical: int
) -> Normalizer:
    if use_log_loss:
        return LogNormalizer(num_features=num_features, num_categorical=num_categorical)
    return Normalizer(num_features=num_features, num_categorical=num_categorical)


def get_target_indices(target: str) -> list[int]:
    targets = {
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
    mode = "weighted" if weighted_loss else "all"
    normalized_graphs = []

    for graph in graphs:
        graph_norm = normalizer.normalize(graph)
        weight = get_weight(
            graph.x[:, 2],
            num_targets,
            mode=mode,
            alpha=alpha,
        )

        # Only weight physical nodes, not virtual nodes
        weight = weight * (graph.x[:, -1] != 1.0).unsqueeze(1).float()
        graph_norm.weight = weight
        normalized_graphs.append(graph_norm)

    return normalized_graphs


def train_model(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    target_indices,
    device,
    num_epochs,
    ms_ids=None,
    ms_edges=None,
):
    model.train()
    loss_history = []
    use_amp = device.type == "cuda"

    for epoch in tqdm(range(num_epochs), dynamic_ncols=True):
        total_loss = 0.0
        total_nodes = 0

        for batch in loader:
            batch = batch.to(device)
            batch.pos = batch.x[:, :3]

            # ms_ids_dev = [ids.to(device) for ids in ms_ids]
            # ms_edges_dev = [es.to(device) for es in ms_edges]

            optimizer.zero_grad()

            y_true = batch.y[:, target_indices]
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                y_pred = model(batch.x, batch.edge_attr, batch)[:, target_indices]
                loss = F.mse_loss(y_pred, y_true, weight=batch.weight)
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN. Check data and model for issues.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        scheduler.step()
        avg_loss = total_loss / total_nodes
        loss_history.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return loss_history


def save_checkpoint(
    output_path: Path,
    model,
    params,
    normalizer,
    args: argparse.Namespace,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": {
                "node_dim": params["node_dim"],
                "edge_dim": params["edge_dim"],
                "output_dim": params["output_dim"],
                "latent_dim": LATENT_DIM,
                "message_passing_steps": args.layers,
                "use_layer_norm": USE_LAYER_NORM,
                "num_categorical": params["num_categorical"],
            },
            "normalizer": normalizer.__class__.__name__,
            "stats": normalizer.stats,
            "training_args": {
                "num_epochs": args.num_epochs,
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


def build_bistride_hierarchy(
    edge_index: torch.Tensor,
    num_physical_nodes: int,
    num_levels: int = 2,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ms_ids = []
    ms_edges = []

    current_ids = torch.arange(num_physical_nodes, device=device)

    for _ in range(num_levels):
        current_ids = current_ids[::2]
        ms_ids.append(current_ids)

        src, dst = edge_index

        # Filter to physical-only edges first
        physical_mask = (src < num_physical_nodes) & (dst < num_physical_nodes)
        phys_src = src[physical_mask]
        phys_dst = dst[physical_mask]

        # Keep only edges where both endpoints survive this level
        id_mask = torch.zeros(num_physical_nodes, dtype=torch.bool, device=device)
        id_mask[current_ids] = True
        surviving = id_mask[phys_src] & id_mask[phys_dst]

        global_src = phys_src[surviving]
        global_dst = phys_dst[surviving]

        # Remap global indices -> local indices (0..len(current_ids)-1)
        global_to_local = torch.full(
            (num_physical_nodes,), -1, dtype=torch.long, device=device
        )
        global_to_local[current_ids] = torch.arange(len(current_ids), device=device)

        local_src = global_to_local[global_src]
        local_dst = global_to_local[global_dst]
        ms_edges.append(torch.stack([local_src, local_dst], dim=0))

    return ms_ids, ms_edges


def main():
    args = parse_args()
    model_name = resolve_model_name(args)

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    graphs, params, dataset_path = load_graphs_and_params(args.dataset)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loaded dataset from: {dataset_path}")

    normalizer = build_normalizer(
        args.log_loss, params["node_dim"], params["num_categorical"]
    )
    normalizer.fit(graphs)

    target_indices = get_target_indices(args.target)
    num_targets = len(target_indices)

    normalized_graphs = prepare_graphs(
        graphs,
        normalizer,
        args.weighted_loss,
        args.alpha,
        num_targets,
    )

    loader = DataLoader(
        normalized_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # ms_ids, ms_edges = build_bistride_hierarchy(
    #     edge_index=normalized_graphs[0].edge_index,
    #     num_physical_nodes=(normalized_graphs[0].x[:, -1] != 1.0).sum().item(),
    #     num_levels=2,
    # )

    model = MeshGraphKAN(
        input_dim_nodes=params["node_dim"],
        input_dim_edges=params["edge_dim"],
        output_dim=params["output_dim"],
        processor_size=args.layers,
        mlp_activation_fn="relu",
        aggregation="sum",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.debug:
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.dtype}, shape: {param.shape}")

    start = time.time()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    loss_history = train_model(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        target_indices=target_indices,
        device=device,
        num_epochs=args.num_epochs,
        # ms_ids=ms_ids,
        # ms_edges=ms_edges,
    )

    checkpoint_path = args.output_dir / f"{model_name}.pth"
    save_checkpoint(checkpoint_path, model, params, normalizer, args)

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

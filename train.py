import argparse
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nets import EncodeProcessDecode
from normalizer import LogNormalizer, Normalizer
from utils import get_weight


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
        "--tensorboard",
        action="store_true",
        help="Whether to enable TensorBoard logging.",
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


def build_normalizer(use_log_loss: bool, node_dim: int):
    if use_log_loss:
        return LogNormalizer(num_features=node_dim)
    return Normalizer(num_features=node_dim)


def get_target_indices(target: str) -> list[int]:
    targets = {
        "all": [0, 1, 2, 3],
        "displacement": [0, 1, 2],
        "stress": [3],
    }
    if target not in targets:
        raise ValueError(f"Unknown target: {target}")
    return targets[target]


def prepare_graphs(graphs, normalizer, weighted_loss: bool, alpha: float, num_targets: int):
    mode = "weighted" if weighted_loss else "all"
    normalized_graphs = []

    for graph in graphs:
        graph_norm = normalizer.normalize(graph)
        graph_norm.weight = get_weight(
            graph.x[:, 2],
            num_targets,
            mode=mode,
            alpha=alpha,
        )
        normalized_graphs.append(graph_norm)

    return normalized_graphs


def create_tensorboard_writer(args: argparse.Namespace, model_name: str):
    if not args.tensorboard:
        return None, None

    log_path = args.log_dir / model_name
    writer = SummaryWriter(log_dir=log_path)
    print(f"TensorBoard logging to: {log_path}")
    return writer, log_path


def train_model(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    target_indices,
    device,
    num_epochs: int,
    writer,
):
    model.train()
    loss_history = []
    use_amp = device.type == "cuda"

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        total_nodes = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_true = batch.y[:, target_indices]
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                y_pred = model(batch)[:, target_indices]
                loss = F.mse_loss(y_pred, y_true, weight=batch.weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        scheduler.step()
        avg_loss = total_loss / total_nodes
        loss_history.append(avg_loss)

        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

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


def main():
    args = parse_args()
    model_name = resolve_model_name(args)

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    graphs, params, dataset_path = load_graphs_and_params(args.dataset)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loaded dataset from: {dataset_path}")

    normalizer = build_normalizer(args.log_loss, params["node_dim"])
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

    model = EncodeProcessDecode(
        node_dim=params["node_dim"],
        edge_dim=params["edge_dim"],
        output_dim=params["output_dim"],
        latent_dim=LATENT_DIM,
        message_passing_steps=args.layers,
        use_layer_norm=USE_LAYER_NORM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.debug:
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.dtype}, shape: {param.shape}")

    writer, log_path = create_tensorboard_writer(args, model_name)

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
        writer=writer,
    )

    checkpoint_path = args.output_dir / f"{model_name}.pth"
    save_checkpoint(checkpoint_path, model, params, normalizer, args)

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")
    print(f"Model saved to: {checkpoint_path}")

    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {log_path}")

    if args.plots:
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()


if __name__ == "__main__":
    main()

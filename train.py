import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nets import EncodeProcessDecode
from normalizer import LogNormalizer, Normalizer
from utils import get_weight, normalize


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")

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


latent_dim = 128
USE_LAYER_NORM = True


def main():
    args = parse_args()

    # Check if args.dataset is a file or a folder
    # If it's a folder, then load every .pt file in the folder
    dataset_path = Path("data") / args.dataset
    if dataset_path.is_dir():
        graphs = []
        for pt_file in dataset_path.glob("*.pt"):
            data = torch.load(pt_file, weights_only=False)
            graphs.extend(data["graphs"])
    else:
        # Load dataset
        dataset_path = Path("data") / f"{args.dataset}.pt"
        data = torch.load(dataset_path, weights_only=False)
        graphs = data["graphs"]

    device = torch.device(args.device)

    if args.log_loss:
        normalizer = LogNormalizer(num_features=data["params"]["node_dim"])
    else:
        normalizer = Normalizer(num_features=data["params"]["node_dim"])
    normalizer.fit(graphs)

    # Exponential scaling factor for mse loss
    alpha = args.alpha
    mode = "weighted" if args.weighted_loss else "all"

    # Loss targets
    if args.target == "all":
        target_indices = list(range(4))
    elif args.target == "displacement":
        target_indices = list(range(3))
    elif args.target == "stress":
        target_indices = [3]
    else:
        raise ValueError(f"Unknown target: {args.target}")

    num_targets = len(target_indices)

    # Precompute weights and attach to each graph
    normalized_graphs = []
    for g in graphs:
        g_norm = normalizer.normalize(g)
        g_norm.weight = get_weight(g.x[:, 2], num_targets, mode=mode, alpha=alpha)
        normalized_graphs.append(g_norm)

    loader = DataLoader(
        normalized_graphs,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = EncodeProcessDecode(
        node_dim=data["params"]["node_dim"],
        edge_dim=data["params"]["edge_dim"],
        output_dim=data["params"]["output_dim"],
        latent_dim=latent_dim,
        message_passing_steps=args.layers,
        use_layer_norm=USE_LAYER_NORM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.debug:
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.dtype}, shape: {param.shape}")

    loss_history = []
    start = time.time()

    # Mixed precision training scaler
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    model.train()
    for epoch in tqdm(range(args.num_epochs)):
        total_loss = 0.0
        total_nodes = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_true = batch.y[:, target_indices]
            with torch.autocast(device_type=args.device, dtype=torch.float16):
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
        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.6f}")

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = (
            f"{args.dataset}_{args.target}_{'w' if args.weighted_loss else 'uw'}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": {
                "node_dim": data["params"]["node_dim"],
                "edge_dim": data["params"]["edge_dim"],
                "output_dim": data["params"]["output_dim"],
                "latent_dim": latent_dim,
                "message_passing_steps": args.layers,
                "use_layer_norm": USE_LAYER_NORM,
            },
            "normalizer": normalizer.__class__.__name__,
            "stats": normalizer.stats,
            "training_args": {
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
            },
        },
        args.output_dir / f"{model_name}.pth",
    )

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")

    if args.plots:
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()


if __name__ == "__main__":
    main()

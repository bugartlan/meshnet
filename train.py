import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from nets import EncodeProcessDecode
from utils import normalize


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/"),
        help="Directory to save the trained model file.",
    )
    p.add_argument(
        "--filename",
        type=str,
        default="model",
        help="Filename for the trained model file.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cantilever_1_10",
        help="Name of the graph dataset file (without extension).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Whether to show the training loss plot.",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Whether to use weighted MSE loss based on distance to the bottom.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Exponential scaling factor for weighted MSE loss.",
    )
    p.add_argument(
        "--num-epochs",
        type=int,
        default=500,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training.",
    )
    p.add_argument(
        "--message-passing-steps",
        type=int,
        default=15,
        help="Number of message passing steps in the model.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print debug information.",
    )
    return p.parse_args()


latent_dim = 128


def main():
    args = parse_args()

    # Load dataset
    dataset_path = Path("data") / f"{args.dataset}.pt"
    data = torch.load(dataset_path, weights_only=False)
    graphs = data["graphs"]

    device = torch.device(args.device)
    train_x = torch.cat([g.x for g in graphs], dim=0)
    train_y = torch.cat([g.y for g in graphs], dim=0)
    train_e = torch.cat([g.edge_attr for g in graphs], dim=0)

    stats = {
        "x_mean": train_x.mean(dim=0),
        "x_std": train_x.std(dim=0),
        "y_mean": train_y.mean(dim=0),
        "y_std": train_y.std(dim=0),
        "e_mean": train_e.mean(dim=0),
        "e_std": train_e.std(dim=0),
    }

    loader = DataLoader(
        [normalize(g, stats) for g in graphs],
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = EncodeProcessDecode(
        node_dim=data["params"]["node_dim"],
        edge_dim=data["params"]["edge_dim"],
        output_dim=data["params"]["output_dim"],
        latent_dim=latent_dim,
        message_passing_steps=args.message_passing_steps,
        use_layer_norm=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.debug:
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.dtype}, shape: {param.shape}")

    # Exponential scaling factor for mse loss
    alpha = 0.1 if args.weighted_loss else 0.0

    loss_history = []
    start = time.time()

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_pred = model(batch)
            y_true = batch.y

            weight = torch.exp(-alpha * batch.x[:, 2].unsqueeze(1))
            weight = weight / weight.mean()

            loss = F.mse_loss(y_pred, y_true, weight=weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
        avg_loss = total_loss / total_nodes
        loss_history.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": {
                "node_dim": data["params"]["node_dim"],
                "edge_dim": data["params"]["edge_dim"],
                "output_dim": data["params"]["output_dim"],
                "latent_dim": latent_dim,
                "message_passing_steps": args.message_passing_steps,
                "use_layer_norm": True,
            },
            "stats": stats,
        },
        args.output_dir / f"{args.filename}.pth",
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

import argparse
from pathlib import Path

import meshio
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from nets import EncodeProcessDecode
from utils import info, msh_to_trimesh, normalize, visualize


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.pth"),
        help="Path to the trained model file.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/cantilever_1_10.pt"),
        help="Path to the graph dataset file.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/"),
        help="Directory to save the visualization plots.",
    )
    p.add_argument(
        "--save-plots",
        action="store_true",
        help="Whether to save the visualization plots to the output directory.",
    )
    p.add_argument(
        "--mesh-name",
        type=str,
        default="cantilever",
        help="Base filename (without extension) for .stl and .msh files.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("meshes"),
        help="Directory containing .stl and .msh.",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Whether to use weighted MSE loss based on node positions.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.save_plots and not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    device = torch.device(args.device)
    print("Using device:", device)

    graphs = torch.load(args.dataset, weights_only=False)

    latent_dim = 128
    node_dim, edge_dim, output_dim = info(graphs[0], debug=True)

    model = EncodeProcessDecode(
        node_dim=node_dim,
        edge_dim=edge_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        message_passing_steps=15,
        use_layer_norm=True,
    ).to(device)

    checkpoint = torch.load(
        args.model_path, map_location=torch.device("cpu"), weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint["stats"]
    loader = DataLoader(
        [normalize(g, stats) for g in graphs], batch_size=1, shuffle=False
    )

    total_loss = 0.0
    total_nodes = 0
    alpha = 0.1  # exponential scaling factor for mse loss

    for batch in loader:
        batch = batch.to(device)

        y_pred = model(batch)
        y_true = batch.y

        if args.weighted_loss:
            weight = torch.exp(-alpha * batch.x[:, 2].unsqueeze(1))
            weight = weight / weight.mean()
            loss = F.mse_loss(y_pred, y_true, weight=weight)
        else:
            weight = torch.ones_like(y_true)

        loss = F.mse_loss(y_pred, y_true, weight=weight)
        loss.backward()

        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
    avg_loss = total_loss / total_nodes
    print(f"Average MSE Loss over dataset: {avg_loss:.6f}")

    if args.save_plots:
        mesh = msh_to_trimesh(meshio.read("meshes/cantilever.msh"))
        for i, g in enumerate(graphs):
            g_pred = g.clone()
            g_pred.y = model(normalize(g, stats).to(device)).detach()
            g_pred.y = g_pred.y * stats["y_std"].to(device) + stats["y_mean"].to(device)

            visualize(
                mesh,
                g,
                force_arrows=True,
                show=False,
                filename=args.output_dir / f"truth_{i}.html",
            )
            visualize(
                mesh,
                g_pred,
                force_arrows=True,
                show=False,
                filename=args.output_dir / f"pred_{i}.html",
            )


if __name__ == "__main__":
    main()

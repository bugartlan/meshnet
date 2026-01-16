import argparse
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from nets import EncodeProcessDecode
from utils import make_pv_mesh, msh_to_trimesh, normalize, visualize_graph


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")
    p.add_argument(
        "--model",
        type=str,
        default="model",
        help="Path to the trained model file.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cantilever_1_10",
        help="Path to the graph dataset file (no extension).",
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
        "-N",
        type=int,
        default=1,
        help="Number of samples to visualize from the dataset.",
    )
    p.add_argument(
        "--save-plots",
        action="store_true",
        help="Whether to save the visualization plots to the output directory.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("meshes/cantilever"),
        help="Directory containing .stl and .msh files.",
    )
    p.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Whether to use weighted MSE loss based on node positions.",
    )
    return p.parse_args()


labels = ["x-displacement", "y-displacement", "z-displacement", "Von Mises Stress"]


def main():
    args = parse_args()

    if args.save_plots and not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    device = torch.device(args.device)
    print("Using device:", device)

    dataset_path = Path("data") / f"{args.dataset}.pt"
    data = torch.load(dataset_path, weights_only=False)
    graphs = data["graphs"]
    meshes = data["meshes"]
    params = data["params"]

    model_path = Path("models") / f"{args.model}.pth"
    checkpoint = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=True
    )
    model_state_dict = checkpoint["model_state_dict"]
    params = checkpoint["params"]
    stats = checkpoint["stats"]
    model = EncodeProcessDecode(
        node_dim=params["node_dim"],
        edge_dim=params["edge_dim"],
        output_dim=params["output_dim"],
        latent_dim=params["latent_dim"],
        message_passing_steps=15,
        use_layer_norm=True,
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

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
        rng = np.random.default_rng(42)
        idx = rng.choice(len(graphs), size=args.N, replace=False)
        for i in range(args.N):
            msh_path = args.input_dir / "msh" / f"{meshes[idx[i]]}.msh"
            mesh = msh_to_trimesh(meshio.read(msh_path))
            g = graphs[idx[i]].clone()
            g.y = model(normalize(g, stats).to(device)).detach()
            g.y = g.y * stats["y_std"].to(device) + stats["y_mean"].to(device)
            pv_mesh_truth = make_pv_mesh(mesh, graphs[idx[i]], labels)
            pv_mesh_pred = make_pv_mesh(mesh, g, labels)
            for j in range(3):
                clim = (pv_mesh_truth[labels[j]].min(), pv_mesh_truth[labels[j]].max())
                visualize_graph(
                    pv_mesh_truth,
                    g,
                    label=labels[j],
                    show=False,
                    force_arrows=True,
                    clim=clim,
                    filename=args.output_dir
                    / f"{meshes[idx[i]]}_sample{i}_true_{labels[j].replace(' ', '_')}.html",
                )
                visualize_graph(
                    pv_mesh_pred,
                    g,
                    label=labels[j],
                    show=False,
                    force_arrows=True,
                    clim=clim,
                    filename=args.output_dir
                    / f"{meshes[idx[i]]}_sample{i}_pred_{labels[j].replace(' ', '_')}.html",
                )


if __name__ == "__main__":
    main()

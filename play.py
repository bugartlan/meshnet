import argparse
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from nets import EncodeProcessDecode
from utils import get_weight, make_pv_mesh, msh_to_trimesh, normalize, visualize_graph


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")

    # --- IO Configuration ---
    p.add_argument(
        "--checkpoint",
        type=str,
        default="model",
        help="Filename of the saved model checkpoint (no extension).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cantilever_1_10",
        help="Name of the graph dataset file (no extension).",
    )
    p.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("meshes/cantilever"),
        help="Directory containing original .msh files. for visualization.",
    )

    # --- Evaluation Configuration ---
    p.add_argument("--mode", choices=["all", "weighted", "bottom"], default="all")
    p.add_argument(
        "--target",
        choices=["all", "displacement", "stress"],
        default="stress",
        help="Which components to include in the loss calculation.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Exponential scaling factor (only used if --mode='weighted').",
    )

    # --- Visualization Configuration ---
    p.add_argument(
        "--plots",
        action="store_true",
        help="Whether to save the visualization plots to the output directory.",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plots/"),
        help="Directory to save the visualization plots.",
    )
    p.add_argument(
        "-n",
        type=int,
        default=1,
        help="Number of samples to visualize from the dataset.",
    )

    # --- Runtime Flags ---
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cpu' or 'cuda').",
    )
    return p.parse_args()


labels = ["x-displacement", "y-displacement", "z-displacement", "Von Mises Stress"]


def main():
    args = parse_args()

    if args.plots and not args.plot_dir.exists():
        args.plot_dir.mkdir(parents=True)

    device = torch.device(args.device)

    dataset_path = Path("data") / f"{args.dataset}.pt"
    data = torch.load(dataset_path, weights_only=False)
    graphs = data["graphs"]
    meshes = data["meshes"]

    model_path = Path("models") / f"{args.checkpoint}.pth"
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
        message_passing_steps=params["message_passing_steps"],
        use_layer_norm=params["use_layer_norm"],
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    loader = DataLoader(
        [normalize(g, stats) for g in graphs], batch_size=1, shuffle=False
    )

    # Loss targets
    if args.target == "all":
        target_indices = list(range(4))
    elif args.target == "displacement":
        target_indices = list(range(3))
    elif args.target == "stress":
        target_indices = [3]
    else:
        raise ValueError(f"Unknown target: {args.target}")

    total_loss = 0.0
    total_nodes = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            y_pred = model(batch)[:, target_indices]
            y_true = batch.y[:, target_indices]

            weight = get_weight(
                batch.x[:, 2],
                y_true.shape[1],
                mode=args.mode,
                alpha=args.alpha,
            )

            loss = F.mse_loss(y_pred, y_true, weight=weight)

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

    avg_loss = total_loss / total_nodes
    print(f"Average MSE Loss over dataset: {avg_loss:.6f}")

    if args.plots:
        print(f"Generating {args.n} plots in {args.plot_dir}...")
        rng = np.random.default_rng(42)

        n_samples = min(args.n, len(graphs))
        idx = rng.choice(len(graphs), size=n_samples, replace=False)

        for i in range(n_samples):
            mesh_name = meshes[idx[i]]
            msh_path = args.mesh_dir / "msh" / f"{mesh_name}.msh"
            mesh = msh_to_trimesh(meshio.read(msh_path))

            g_true = graphs[idx[i]]
            g_input = normalize(g_true, stats).to(device)

            # Predict
            with torch.no_grad():
                pred_normalized = model(g_input)

            # Denormalize predictions to get physical values
            y_pred_phys = pred_normalized * stats["y_std"].to(device) + stats[
                "y_mean"
            ].to(device)

            # Create graphs for visualization
            g_pred = g_true.clone()
            g_pred.y = y_pred_phys.cpu()

            pv_mesh_true = make_pv_mesh(mesh, g_true, labels)
            pv_mesh_pred = make_pv_mesh(mesh, g_pred, labels)

            for j in target_indices:
                label_name = labels[j]
                safe_label = label_name.replace(" ", "_")

                # Determine color limits based on true values
                clim = (pv_mesh_true[labels[j]].min(), pv_mesh_true[labels[j]].max())
                filename_true = (
                    args.plot_dir / f"{mesh_name}_sample{i}_true_{safe_label}.html"
                )
                filename_pred = (
                    args.plot_dir / f"{mesh_name}_sample{i}_pred_{safe_label}.html"
                )

                # Plot ground truth and prediction
                visualize_graph(
                    pv_mesh_true,
                    g_true,
                    label=labels[j],
                    show=False,
                    force_arrows=True,
                    clim=clim,
                    filename=filename_true,
                )
                visualize_graph(
                    pv_mesh_pred,
                    g_pred,
                    label=labels[j],
                    show=False,
                    force_arrows=True,
                    clim=clim,
                    filename=filename_pred,
                )

            print(f"Saved plots for sample {i} ({mesh_name}).")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from nets import EncodeProcessDecode
from utils import (
    get_weight,
    make_pv_mesh,
    msh_to_trimesh,
    normalize,
    strain_stress_vm,
    visualize_graph,
)

################################ Material Properties ###################################
E = 2.0e9  # Young's modulus
nu = 0.35  # Poisson's ratio
#########################################################################################


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
        "--compute_stress",
        action="store_true",
        help="If set, compute von Mises stress from the predicted displacement.",
    )
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
    graphs = [g.to(device) for g in data["graphs"]]
    mesh_name = data["mesh"]

    model_path = Path("models") / f"{args.checkpoint}.pth"
    checkpoint = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=True
    )
    model_state_dict = checkpoint["model_state_dict"]
    params = checkpoint["params"]
    stats = {k: v.to(device) for k, v in checkpoint["stats"].items()}
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

    graphs_pred = []
    with torch.no_grad():
        for g in graphs:
            normalized_g = normalize(g, stats)

            y_pred = model(normalized_g)
            y_pred = y_pred * stats["y_std"] + stats["y_mean"]

            g_pred = g.clone()
            g_pred.y = y_pred

            if args.compute_stress:
                eps, sigma, vm = strain_stress_vm(g_pred, E, nu)
                g_pred.y[:, 3] = vm

            graphs_pred.append(g_pred)

            y_true = g.y[:, target_indices]
            y_pred = g_pred.y[:, target_indices]

            weight = get_weight(
                g.x[:, 2],
                y_true.shape[1],
                mode=args.mode,
                alpha=args.alpha,
            )

            loss = F.l1_loss(y_pred, y_true, weight=weight)

            total_loss += loss.item() * g.num_nodes
            total_nodes += g.num_nodes

    avg_loss = total_loss / total_nodes
    print(f"Average L1 Loss over dataset: {avg_loss:.6f}")

    if args.plots:
        print(f"Generating {args.n} plots in {args.plot_dir}...")
        rng = np.random.default_rng(42)

        n_samples = min(args.n, len(graphs_pred))
        idx = rng.choice(len(graphs_pred), size=n_samples, replace=False)

        is_coarse = args.dataset.endswith("_c")

        for i in range(n_samples):
            msh_path = (
                args.mesh_dir
                / ("msh_coarse" if is_coarse else "msh")
                / f"{mesh_name}.msh"
            )
            mesh = msh_to_trimesh(meshio.read(msh_path))

            g_true = graphs[idx[i]].cpu()
            g_pred = graphs_pred[idx[i]].cpu()

            pv_mesh_true = make_pv_mesh(mesh, g_true, labels)
            pv_mesh_pred = make_pv_mesh(mesh, g_pred, labels)

            for j in target_indices:
                label_name = labels[j].replace(" ", "_")

                # Determine color limits based on true values
                clim = (pv_mesh_true[labels[j]].min(), pv_mesh_true[labels[j]].max())
                filename_true = (
                    args.plot_dir / f"{mesh_name}_sample{i}_true_{label_name}.html"
                )
                filename_pred = (
                    args.plot_dir / f"{mesh_name}_sample{i}_pred_{label_name}.html"
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

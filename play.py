import argparse
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F

from graph_builder import GraphVisualizer
from nets import EncodeProcessDecode
from normalizer import LogNormalizer, Normalizer
from utils import (
    get_weight,
    msh_to_trimesh,
    strain_stress_vm,
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

    data = torch.load(f"data/{args.dataset}.pt", weights_only=False)
    graphs = [g.to(device) for g in data["graphs"]]
    msh_path = data["mesh"]

    checkpoint = torch.load(
        f"models/{args.checkpoint}.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model_state_dict = checkpoint["model_state_dict"]
    params = checkpoint["params"]

    normalizer = LogNormalizer(
        num_features=params["node_dim"], device=device, stats=checkpoint["stats"]
    )
    if checkpoint["normalizer"] == "LogNormalizer":
        normalizer = LogNormalizer(
            num_features=params["node_dim"], device=device, stats=checkpoint["stats"]
        )
    else:
        normalizer = Normalizer(
            num_features=params["node_dim"], device=device, stats=checkpoint["stats"]
        )

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
            normalized_g = normalizer.normalize(g)

            y_pred = model(normalized_g)
            y_pred = normalizer.denormalize_y(y_pred)

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
        visualizer = GraphVisualizer(
            msh_to_trimesh(meshio.read(msh_path)), jupyter_backend=False
        )

        suffix = "_bottom" if args.mode == "bottom" else ""

        for i in range(n_samples):
            g_true = graphs[idx[i]].cpu()
            g_pred = graphs_pred[idx[i]].cpu()

            filename_true = args.plot_dir / f"{msh_path.stem}_smpl{i}_true{suffix}.html"
            filename_pred = args.plot_dir / f"{msh_path.stem}_smpl{i}_pred{suffix}.html"

            # Visualize ground truth and prediction
            if args.mode == "bottom":
                visualizer.bottom(g_true, save_path=filename_true)
                visualizer.bottom(g_pred, save_path=filename_pred)
            else:
                visualizer.stress(g_true, save_path=filename_true)
                visualizer.stress(g_pred, save_path=filename_pred)

            print(f"Saved plots for sample {i} ({msh_path.stem}).")


if __name__ == "__main__":
    main()

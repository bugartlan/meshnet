import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F

from graph_builder import GraphVisualizer
from nets import EncodeProcessDecode, MeshGraphNet
from normalizer import LogNormalizer, Normalizer
from utils import get_weight, msh_to_trimesh


def prepare_graphs(graphs, normalizer):
    normalized_graphs = []

    for graph in graphs:
        graph_norm = normalizer.normalize(graph)
        weight = get_weight(graph.x[:, 2], 1, mode="bottom")

        # Only weight physical nodes, not virtual nodes
        weight = weight * (graph.x[:, -1] != 1.0).unsqueeze(1).float()
        graph_norm.weight = weight
        graph_norm.y = graph.y
        normalized_graphs.append(graph_norm)

    return normalized_graphs


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
    print(f"Loaded dataset '{args.dataset}' with {len(graphs)} graphs.")
    print(
        f"Each graph has {graphs[0].num_nodes} nodes and {graphs[0].num_edges} edges."
    )

    msh_path = data["mesh"]

    checkpoint = torch.load(
        f"models/{args.checkpoint}.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model_state_dict = checkpoint["model_state_dict"]
    params = checkpoint["params"]
    print(f"Loaded model checkpoint '{args.checkpoint}' with parameters:")
    print(f"    Node dim: {params['node_dim']}")
    print(f"    Edge dim: {params['edge_dim']}")
    print(f"    Output dim: {params['output_dim']}")

    if checkpoint["normalizer"] == "LogNormalizer":
        normalizer = LogNormalizer(
            num_features=params["node_dim"],
            num_categorical=params["num_categorical"],
            device=device,
            stats=checkpoint["stats"],
        )
    else:
        normalizer = Normalizer(
            num_features=params["node_dim"],
            num_categorical=params["num_categorical"],
            device=device,
            stats=checkpoint["stats"],
        )

    # Pre-normalize graphs
    normalized_graphs = prepare_graphs(graphs, normalizer)

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
        start = time.time()
        for g in normalized_graphs:
            y_pred = model(g)
            y_pred = normalizer.denormalize_y(y_pred)

            g_pred = g.clone()
            g_pred.y = y_pred

            graphs_pred.append(g_pred)

            y_true = g.y[:, target_indices]
            y_pred = g_pred.y[:, target_indices]

            loss = F.l1_loss(y_pred, y_true, weight=g.weight)

            num_nodes = g.weight.sum().item()
            total_loss += loss.item() * num_nodes
            total_nodes += num_nodes
        end = time.time()
        print(f"Inference completed in {end - start:.2f} seconds.")

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

            n_phys = g_true.num_physical_nodes

            # Create error graph for von Mises stress visualization
            g_err_vm = g_true.clone()
            g_err_vm.y = g_true.y.clone()
            g_err_vm.y[:n_phys, 3] = (g_pred.y[:n_phys, 3] - g_true.y[:n_phys, 3]).abs()

            filename_true = args.plot_dir / f"{msh_path.stem}_smpl{i}_true{suffix}.html"
            filename_pred = args.plot_dir / f"{msh_path.stem}_smpl{i}_pred{suffix}.html"
            filename_err = args.plot_dir / f"{msh_path.stem}_smpl{i}_error{suffix}.html"

            # Visualize ground truth and prediction
            if args.mode == "bottom":
                bottom_mask = torch.isclose(
                    g_true.x[:n_phys, 2],
                    torch.zeros_like(g_true.x[:n_phys, 2]),
                    atol=1e-6,
                )
                true_bottom = g_true.y[:n_phys, 3][bottom_mask]
                pred_bottom = g_pred.y[:n_phys, 3][bottom_mask]
                clim = (
                    torch.min(torch.cat([true_bottom, pred_bottom])).item(),
                    torch.max(torch.cat([true_bottom, pred_bottom])).item(),
                )
                visualizer.bottom(g_true, clim=clim, save_path=filename_true)
                visualizer.bottom(g_pred, clim=clim, save_path=filename_pred)
                visualizer.bottom(g_err_vm, clim=clim, save_path=filename_err)

            else:
                visualizer.stress(g_true, save_path=filename_true)
                visualizer.stress(g_pred, save_path=filename_pred)

            print(f"Saved plots for sample {idx[i]} ({msh_path.stem}).")


if __name__ == "__main__":
    main()

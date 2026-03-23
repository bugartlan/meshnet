import argparse
import time
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

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


def plot(g1, g2, visualizer, mode, f1, f2, f3):
    n_phys = g1.num_physical_nodes

    # Create error graph for von Mises stress visualization
    g_err_vm = g1.clone()
    g_err_vm.y = g1.y.clone()
    g_err_vm.y[:n_phys, 3] = (g2.y[:n_phys, 3] - g1.y[:n_phys, 3]).abs()

    # Visualize ground truth and prediction
    if mode == "bottom":
        bottom_mask = torch.isclose(
            g1.x[:n_phys, 2],
            torch.zeros_like(g1.x[:n_phys, 2]),
            atol=1e-6,
        )
        true_bottom = g1.y[:n_phys, 3][bottom_mask]
        pred_bottom = g2.y[:n_phys, 3][bottom_mask]
        clim = (
            torch.min(torch.cat([true_bottom, pred_bottom])).item(),
            torch.max(torch.cat([true_bottom, pred_bottom])).item(),
        )
        visualizer.bottom(g1, clim=clim, save_path=f1)
        visualizer.bottom(g2, clim=clim, save_path=f2)
        visualizer.bottom(g_err_vm, clim=clim, save_path=f3)

    else:
        clim = (
            torch.min(torch.cat([g1.y[:n_phys, 3], g2.y[:n_phys, 3]])).item(),
            torch.max(torch.cat([g1.y[:n_phys, 3], g2.y[:n_phys, 3]])).item(),
        )
        visualizer.stress(g1, clim=clim, save_path=f1)
        visualizer.stress(g2, clim=clim, save_path=f2)
        visualizer.stress(g_err_vm, clim=clim, save_path=f3)


def mae75(x: np.ndarray, y: np.ndarray, weight: np.ndarray = None) -> float:
    x = x[weight > 0]
    y = y[weight > 0]
    mask = y >= np.percentile(y, 75)
    return np.abs(x - y)[mask].mean()


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

    num_nodes = graphs[0].num_nodes
    num_edges = graphs[0].num_edges
    print(f"Each graph has {num_nodes} nodes and {num_edges} edges.")

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
    total_loss75 = 0.0
    total_kendall = 0.0

    graphs_pred = []
    with torch.no_grad():
        start = time.time()
        for g in normalized_graphs:
            y_pred = model(g)
            y_pred = normalizer.denormalize_y(y_pred)

            g_pred = g.clone()
            g_pred.y = y_pred

            y_true = g.y[:, target_indices]
            y_pred = g_pred.y[:, target_indices]

            loss = F.l1_loss(y_pred, y_true, weight=g.weight).item()
            loss75 = mae75(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                weight=g.weight.cpu().numpy(),
            )
            tau = kendalltau(y_true.cpu().numpy(), y_pred.cpu().numpy()).statistic
            total_loss += loss
            total_loss75 += loss75
            total_kendall += tau

            graphs_pred.append((g_pred.cpu(), tau, loss, loss75))
        end = time.time()
        print(f"Inference completed in {end - start:.2f} seconds.")

    print("Results:")
    print(f"Average L1 Loss over dataset: {total_loss / len(graphs):.6f}")
    print(f"Average L1 Loss (75th percentile): {total_loss75 / len(graphs):.6f}")
    print(f"Average Kendall's Tau metric: {total_kendall / len(graphs):.4f}")
    min_tau_idx, (min_tau_g, min_tau, min_tau_loss, min_tau_loss75) = min(
        enumerate(graphs_pred), key=lambda x: x[1][1]
    )
    max_tau_idx, (max_tau_g, max_tau, max_tau_loss, max_tau_loss75) = max(
        enumerate(graphs_pred), key=lambda x: x[1][1]
    )
    print(f"Min Tau: {min_tau:.4f}, Max Tau: {max_tau:.4f}")

    if args.plots:
        rng = np.random.default_rng(42)
        idx = rng.choice(
            len(graphs_pred), size=min(args.n, len(graphs_pred)), replace=False
        )
        visualizer = GraphVisualizer(
            msh_to_trimesh(meshio.read(msh_path)), jupyter_backend=False
        )

        suffix = "_bottom" if args.mode == "bottom" else ""

        plot(
            graphs[min_tau_idx].cpu(),
            min_tau_g,
            visualizer,
            args.mode,
            args.plot_dir / f"{msh_path.stem}_min_tau_true{suffix}.html",
            args.plot_dir / f"{msh_path.stem}_min_tau_pred{suffix}.html",
            args.plot_dir / f"{msh_path.stem}_min_tau_error{suffix}.html",
        )
        print(
            f"Saved plots for min Tau sample {min_tau_idx} ({msh_path.stem}): "
            f"tau={min_tau:.4f}, loss={min_tau_loss:.6f}, "
            f"loss (75%)={min_tau_loss75:.6f}."
        )
        plot(
            graphs[max_tau_idx].cpu(),
            max_tau_g,
            visualizer,
            args.mode,
            args.plot_dir / f"{msh_path.stem}_max_tau_true{suffix}.html",
            args.plot_dir / f"{msh_path.stem}_max_tau_pred{suffix}.html",
            args.plot_dir / f"{msh_path.stem}_max_tau_error{suffix}.html",
        )
        print(
            f"Saved plots for max Tau sample {max_tau_idx} ({msh_path.stem}): "
            f"tau={max_tau:.4f}, loss={max_tau_loss:.6f}, "
            f"loss (75%)={max_tau_loss75:.6f}."
        )

        print(f"Generating {len(idx)} plots in {args.plot_dir}...")
        for i in idx:
            g_true = graphs[i].cpu()
            g_pred, tau, loss, loss75 = graphs_pred[i]

            filename_true = args.plot_dir / f"{msh_path.stem}_smpl{i}_true{suffix}.html"
            filename_pred = args.plot_dir / f"{msh_path.stem}_smpl{i}_pred{suffix}.html"
            filename_err = args.plot_dir / f"{msh_path.stem}_smpl{i}_error{suffix}.html"

            plot(
                g_true,
                g_pred,
                visualizer,
                args.mode,
                filename_true,
                filename_pred,
                filename_err,
            )

            print(
                f"Saved plots for sample {i} ({msh_path.stem}): "
                f"tau={tau:.4f}, loss={loss:.6f}, "
                f"loss (75%)={loss75:.6f}."
            )


if __name__ == "__main__":
    main()

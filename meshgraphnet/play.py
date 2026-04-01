import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F
from graph_builder import GraphVisualizer
from nets import EncodeProcessDecode, MeshGraphNet
from normalizer import LogNormalizer, Normalizer
from scipy.stats import kendalltau
from utils import get_weight, msh_to_trimesh

LABELS = ["x-displacement", "y-displacement", "z-displacement", "Von Mises Stress"]


@dataclass
class PredictionResult:
    """Stores the predicted graph and its evaluation metrics for one sample."""

    graph: object
    kendall_tau: float
    loss: float
    loss75: float


@dataclass
class PlotPaths:
    """File paths for the three output plots of a single sample (true / pred / error)."""

    true: Path
    pred: Path
    error: Path


# ---------------------------------------------------------------------------
# Graph preparation
# ---------------------------------------------------------------------------


def prepare_graphs(graphs: list, normalizer: Normalizer, mode: str = "bottom"):
    """Normalize graphs and attach per-node importance weights.

    Args:
        graphs: Raw graph objects to normalise.
        normalizer: Fitted normalizer instance.
        mode: Weighting mode passed to ``get_weight``.

    Returns:
        List of normalised graph objects with a ``weight`` attribute.
    """
    normalized_graphs = []

    for graph in graphs:
        graph_norm = normalizer.normalize(graph)

        weight = get_weight(graph.x[:, 2], 1, mode=mode)
        # Only weight physical nodes, not virtual nodes
        weight = weight * (graph.x[:, -1] != 1.0).unsqueeze(1).float()

        graph_norm.weight = weight
        graph_norm.y = graph.y
        normalized_graphs.append(graph_norm)

    return normalized_graphs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae75(pred: np.ndarray, true: np.ndarray, weight: np.ndarray) -> float:
    """Mean absolute error restricted to nodes above the 75th-percentile of *true*.

    Args:
        pred: Predicted values.
        true: Ground-truth values.
        weight: Per-node weights; nodes with ``weight <= 0`` are excluded.

    Returns:
        Scalar MAE over the top-25 % of ground-truth values.
    """
    x = pred[weight > 0]
    y = true[weight > 0]
    mask = y >= np.percentile(y, 75)
    return np.abs(x - y)[mask].mean()


def compute_kendall_tau(
    pred: np.ndarray, true: np.ndarray, weight: np.ndarray
) -> float:
    """Kendall's τ rank-correlation between *pred* and *true*.

    Args:
        pred: Predicted values.
        true: Ground-truth values.
        weight: Per-node weights; nodes with ``weight <= 0`` are excluded.

    Returns:
        Kendall's τ statistic.
    """
    x = pred[weight > 0]
    y = true[weight > 0]
    return kendalltau(x, y).statistic


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot(g_true, g_pred, visualizer: GraphVisualizer, mode: str, paths: PlotPaths):
    """Render and save ground-truth, prediction, and absolute-error plots.

    Args:
        g_true: Graph carrying ground-truth labels.
        g_pred: Graph carrying predicted labels.
        visualizer: ``GraphVisualizer`` instance bound to the mesh.
        mode: ``"bottom"`` to render the bottom surface only, else full mesh.
        paths: Output file paths for the three plot files.
    """
    n_phys = g_true.num_physical_nodes

    # Create error graph for von Mises stress visualization
    g_err = g_true.clone()
    g_err.y = g_true.y.clone()
    g_err.y[:n_phys, 3] = (g_pred.y[:n_phys, 3] - g_true.y[:n_phys, 3]).abs()

    if mode == "bottom":
        bottom_mask = torch.isclose(
            g_true.x[:n_phys, 2],
            torch.zeros_like(g_true.x[:n_phys, 2]),
            atol=1e-6,
        )
        true_vals = g_true.y[:n_phys, 3][bottom_mask]
        pred_vals = g_pred.y[:n_phys, 3][bottom_mask]
        clim = (
            torch.min(torch.cat([true_vals, pred_vals])).item(),
            torch.max(torch.cat([true_vals, pred_vals])).item(),
        )
        visualizer.bottom(g_true, clim=clim, save_path=paths.true)
        visualizer.bottom(g_pred, clim=clim, save_path=paths.pred)
        visualizer.bottom(g_err, clim=clim, save_path=paths.error)
    else:
        clim = (
            torch.min(torch.cat([g_true.y[:n_phys, 3], g_pred.y[:n_phys, 3]])).item(),
            torch.max(torch.cat([g_true.y[:n_phys, 3], g_pred.y[:n_phys, 3]])).item(),
        )
        visualizer.stress(g_true, clim=clim, save_path=paths.true)
        visualizer.stress(g_pred, clim=clim, save_path=paths.pred)
        visualizer.stress(g_err, clim=clim, save_path=paths.error)


# ---------------------------------------------------------------------------
# Model / normalizer helpers
# ---------------------------------------------------------------------------


def build_normalizer(checkpoint: dict, device: torch.device):
    """Reconstruct the normalizer from a model checkpoint.

    Args:
        checkpoint: Dict loaded from a ``.pth`` checkpoint file.
        device: Target compute device.

    Returns:
        A fitted ``Normalizer`` or ``LogNormalizer`` instance.
    """
    params = checkpoint["params"]
    stats = checkpoint["stats"]
    kwargs = dict(
        num_features=params["node_dim"],
        num_categorical=params["num_categorical"],
        device=device,
        stats=stats,
    )
    if checkpoint["normalizer"] == "LogNormalizer":
        return LogNormalizer(**kwargs)
    return Normalizer(**kwargs)


def get_target_indices(target: str) -> list[int]:
    """Map a target name to the corresponding output column indices.

    Args:
        target: One of ``"all"``, ``"displacement"``, or ``"stress"``.

    Returns:
        List of integer column indices.

    Raises:
        ValueError: If *target* is not recognised.
    """
    match target:
        case "all":
            return list(range(4))
        case "displacement":
            return list(range(3))
        case "stress":
            return [3]
        case _:
            raise ValueError(f"Unknown target: {target!r}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: torch.nn.Module,
    normalized_graphs: list,
    normalizer,
    target_indices: list[int],
) -> tuple[list[PredictionResult], float]:
    """Run the model over all graphs and collect per-sample metrics.

    Args:
        model: Trained ``EncodeProcessDecode`` model in eval mode.
        normalized_graphs: Pre-normalised graphs (with ``weight`` attribute).
        normalizer: Normalizer used to invert the model's output scale.
        target_indices: Output columns to include in loss computation.

    Returns:
        Tuple of (list of ``PredictionResult``, elapsed seconds).
    """
    results: list[PredictionResult] = []

    start = time.time()
    with torch.no_grad():
        for g in normalized_graphs:
            y_pred = model(g)
            y_pred = normalizer.denormalize_y(y_pred)

            g_pred = g.clone()
            g_pred.y = y_pred

            y_true = g.y[:, target_indices]
            y_pred = g_pred.y[:, target_indices]
            weight_np = g.weight.cpu().numpy()

            loss = F.l1_loss(y_pred, y_true, weight=g.weight).item()
            loss75 = mae75(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                weight=weight_np,
            )
            tau = compute_kendall_tau(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                weight=weight_np,
            )

            results.append(
                PredictionResult(
                    graph=g_pred.cpu(), kendall_tau=tau, loss=loss, loss75=loss75
                )
            )
    elapsed = time.time() - start

    return results, elapsed


# ---------------------------------------------------------------------------
# Plot saving
# ---------------------------------------------------------------------------


def save_plots(
    results: list[PredictionResult],
    graphs_original: list,
    msh_path: Path,
    visualizer: GraphVisualizer,
    mode: str,
    plot_dir: Path,
    n_random: int,
) -> None:
    """Save plots for the best/worst τ samples and *n_random* random samples.

    Args:
        results: Per-sample prediction results from ``run_inference``.
        graphs_original: Un-normalised ground-truth graphs (CPU tensors).
        msh_path: Path to the source mesh file (stem used in filenames).
        visualizer: ``GraphVisualizer`` bound to the loaded mesh.
        mode: ``"bottom"`` or ``"all"``.
        plot_dir: Directory where HTML plot files are written.
        n_random: Number of random samples to visualise in addition to extremes.
    """
    suffix = "_bottom" if mode == "bottom" else ""

    def _make_paths(tag: str) -> PlotPaths:
        return PlotPaths(
            true=plot_dir / f"{msh_path.stem}_{tag}_true{suffix}.html",
            pred=plot_dir / f"{msh_path.stem}_{tag}_pred{suffix}.html",
            error=plot_dir / f"{msh_path.stem}_{tag}_error{suffix}.html",
        )

    def _plot_and_log(idx: int, tag: str) -> None:
        result = results[idx]
        plot(
            graphs_original[idx].cpu(), result.graph, visualizer, mode, _make_paths(tag)
        )
        print(
            f"Saved plots for {tag} (sample {idx}, {msh_path.stem}): "
            f"tau={result.kendall_tau:.4f}, loss={result.loss:.6f}, "
            f"loss75={result.loss75:.6f}."
        )

    min_idx = min(range(len(results)), key=lambda i: results[i].kendall_tau)
    max_idx = max(range(len(results)), key=lambda i: results[i].kendall_tau)
    _plot_and_log(min_idx, "min_tau")
    _plot_and_log(max_idx, "max_tau")

    rng = np.random.default_rng(42)
    random_indices = rng.choice(
        len(results), size=min(n_random, len(results)), replace=False
    )
    print(f"Generating {len(random_indices)} random plots in {plot_dir}...")
    for i in random_indices:
        _plot_and_log(i, f"smpl{i}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate trained model on a dataset.")

    p.add_argument(
        "--checkpoint",
        type=str,
        default="model",
        help="Filename of the saved model checkpoint (no extension).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Name of the graph dataset file (no extension).",
    )
    p.add_argument("--mode", choices=["all", "bottom"], default="all")
    p.add_argument(
        "--target",
        choices=["all", "displacement", "stress"],
        default="stress",
        help="Which components to include in the loss calculation.",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Save visualisation plots to the output directory.",
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
        help="Number of random samples to visualise.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (e.g. 'cpu' or 'cuda').",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.plots:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load dataset
    data = torch.load(f"data/{args.dataset}.pt", weights_only=False)
    graphs = [g.to(device) for g in data["graphs"]]
    msh_path: Path = data["mesh"]
    print(f"Loaded dataset '{args.dataset}' with {len(graphs)} graphs.")
    print(
        f"Each graph has {graphs[0].num_nodes} nodes and {graphs[0].num_edges} edges."
    )

    # Load checkpoint
    checkpoint = torch.load(
        f"models/{args.checkpoint}.pth", map_location="cpu", weights_only=True
    )
    params = checkpoint["params"]
    print(
        f"Loaded checkpoint '{args.checkpoint}' — "
        f"node_dim={params['node_dim']}, "
        f"edge_dim={params['edge_dim']}, "
        f"output_dim={params['output_dim']}."
    )

    normalizer = build_normalizer(checkpoint, device)
    normalized_graphs = prepare_graphs(graphs, normalizer, args.mode)
    target_indices = get_target_indices(args.target)

    model = EncodeProcessDecode(
        node_dim=params["node_dim"],
        edge_dim=params["edge_dim"],
        output_dim=params["output_dim"],
        latent_dim=params["latent_dim"],
        message_passing_steps=params["message_passing_steps"],
        use_layer_norm=params["use_layer_norm"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    results, elapsed = run_inference(
        model, normalized_graphs, normalizer, target_indices
    )
    print(f"Inference completed in {elapsed:.2f}s.")

    # Aggregate metrics
    n = len(results)
    avg_loss = sum(r.loss for r in results) / n
    avg_loss75 = sum(r.loss75 for r in results) / n
    avg_tau = sum(r.kendall_tau for r in results) / n
    min_tau = min(r.kendall_tau for r in results)
    max_tau = max(r.kendall_tau for r in results)

    print("Results:")
    print(f"  Avg L1 loss:              {avg_loss:.6f}")
    print(f"  Avg L1 loss (75th pct):   {avg_loss75:.6f}")
    print(f"  Avg Kendall's τ:          {avg_tau:.4f}")
    print(f"  Min / Max τ:              {min_tau:.4f} / {max_tau:.4f}")

    # Optionally save visualisation plots
    if args.plots:
        visualizer = GraphVisualizer(
            msh_to_trimesh(meshio.read(msh_path)), jupyter_backend=False
        )
        save_plots(
            results, graphs, msh_path, visualizer, args.mode, args.plot_dir, args.n
        )


if __name__ == "__main__":
    main()

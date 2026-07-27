import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F

from meshnet.mgn.graphs import GraphVisualizer
from meshnet.mgn.nets import EncodeProcessDecode, MeshGraphNet
from meshnet.mgn.normalizer import LogNormalizer, Normalizer
from meshnet.mgn.utils import msh_to_trimesh

LABELS = ["x-displacement", "y-displacement", "z-displacement", "Von Mises Stress"]


@dataclass
class PredictionResult:
    """Stores the predicted graph and its evaluation metrics for one sample."""

    graph: object
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
    """Normalize graphs and create loss masks.

    Args:
        graphs: Raw graph objects to normalise.
        normalizer: Fitted normalizer instance.
        mode: `bottom` to include only the bottom surface in the loss, `all` to include all nodes.

    Returns:
        List of normalised graph objects with a ``loss_mask`` attribute.
    """
    normalized_graphs = []

    for graph in graphs:
        graph_norm = normalizer.normalize(graph)

        num_phys = int(graph.num_physical_nodes)
        loss_mask = torch.zeros_like(graph.x[:, 0], dtype=torch.bool)
        loss_mask[:num_phys] = mode != "bottom" or graph.x[:num_phys, 2] <= 1e-4

        graph_norm.loss_mask = loss_mask
        graph_norm.y = graph.y
        normalized_graphs.append(graph_norm)

    return normalized_graphs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae75(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error restricted to nodes above the 75th-percentile of *true*.

    Args:
        pred: Predicted values.
        true: Ground-truth values.

    Returns:
        Scalar MAE over the top-25 % of ground-truth values.
    """
    top = true >= np.percentile(true, 75)
    return np.abs(pred[top] - true[top]).mean()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _build_stress_error_graph(g_true, g_pred, n_phys: int):
    """Create a graph whose stress channel stores absolute prediction error."""
    g_err = g_true.clone()
    g_err.y = g_true.y.clone()
    g_err.y[:n_phys, 3] = (g_pred.y[:n_phys, 3] - g_true.y[:n_phys, 3]).abs()
    return g_err


def _stress_values(graph, n_phys: int, mode: str) -> torch.Tensor:
    """Extract stress values used for shared color scaling."""
    stress = graph.y[:n_phys, 3]
    if mode != "bottom":
        return stress

    z = graph.x[:n_phys, 2]
    bottom_mask = torch.isclose(z, torch.zeros_like(z), atol=1e-6)
    # Fallback to all physical nodes when a strict bottom slice is empty.
    if not torch.any(bottom_mask):
        return stress
    return stress[bottom_mask]


def _stress_clim(g_true, g_pred, n_phys: int, mode: str) -> tuple[float, float]:
    """Compute color limits shared by true and predicted stress fields."""
    vals = torch.cat(
        [
            _stress_values(g_true, n_phys=n_phys, mode=mode),
            _stress_values(g_pred, n_phys=n_phys, mode=mode),
        ]
    )
    return (vals.min().item(), vals.max().item())


def plot(g_true, g_pred, visualizer: GraphVisualizer, mode: str, paths: PlotPaths):
    """Render and save ground-truth, prediction, and absolute-error plots.

    Args:
        g_true: Graph carrying ground-truth labels.
        g_pred: Graph carrying predicted labels.
        visualizer: ``GraphVisualizer`` instance bound to the mesh.
        mode: ``"bottom"`` to render the bottom surface only, else full mesh.
        paths: Output file paths for the three plot files.
    """
    if mode not in {"bottom", "all"}:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected 'bottom' or 'all'.")

    n_phys = g_true.num_physical_nodes
    g_err = _build_stress_error_graph(g_true, g_pred, n_phys=n_phys)
    clim = _stress_clim(g_true, g_pred, n_phys=n_phys, mode=mode)
    render = visualizer.bottom if mode == "bottom" else visualizer.stress

    for graph, out_path in (
        (g_true, paths.true),
        (g_pred, paths.pred),
        (g_err, paths.error),
    ):
        render(graph, clim=clim, save_path=out_path)


def render_stress(g_true, g_pred, visualizer: GraphVisualizer, paths: PlotPaths):
    """Render and save ground-truth, prediction, and absolute-error plots.

    Args:
        g_true: Graph carrying ground-truth labels.
        g_pred: Graph carrying predicted labels.
        visualizer: ``GraphVisualizer`` instance bound to the mesh.
        paths: Output file paths for the three plot files.
    """
    for graph, out_path in (
        (g_true, paths.true),
        (g_pred, paths.pred),
    ):
        visualizer.stress(graph, save_path=out_path)


def render_displacement(g_true, g_pred, visualizer: GraphVisualizer, paths: PlotPaths):
    """Render and save ground-truth, prediction, and absolute-error plots.

    Args:
        g_true: Graph carrying ground-truth labels.
        g_pred: Graph carrying predicted labels.
        visualizer: ``GraphVisualizer`` instance bound to the mesh.
        paths: Output file paths for the three plot files.
    """
    for graph, out_path in (
        (g_true, paths.true),
        (g_pred, paths.pred),
    ):
        visualizer.displacement(graph, save_path=out_path)


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
    kwargs = {
        "num_features": params["node_dim"],
        "num_categorical": params["num_categorical"],
        "device": device,
        "stats": stats,
    }
    if checkpoint["normalizer"] == "LogNormalizer":
        return LogNormalizer(**kwargs)
    return Normalizer(**kwargs)


def get_target_indices(target: str) -> list[int]:
    """Map a target name to the corresponding output column indices.

    Args:
        target: One of ``"all"``, ``"displacement"``, or ``"stress"``.

    Returns:
        List of integer column indices.
    """
    match target:
        case "all":
            return list(range(9))
        case "displacement":
            return list(range(3))
        case "stress":
            return list(range(3, 9))
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

            loss_mask = g.loss_mask
            y_true = g.y[loss_mask][:, target_indices]
            y_pred = g_pred.y[loss_mask][:, target_indices]

            loss = F.l1_loss(y_pred, y_true).item()
            loss75 = mae75(y_pred.cpu().numpy(), y_true.cpu().numpy())

            results.append(
                PredictionResult(graph=g_pred.cpu(), loss=loss, loss75=loss75)
            )
    elapsed = time.time() - start

    return results, elapsed


# ---------------------------------------------------------------------------
# Plot saving
# ---------------------------------------------------------------------------


def save_plots(
    results: list[PredictionResult],
    graphs: list,
    filepath: Path,
    visualizer: GraphVisualizer,
    mode: str,
    directory: Path,
    n_random: int = 1,
    fields: str = "displacement",
) -> None:
    """Save plots for *n_random* random samples.

    Args:
        results: Per-sample prediction results from ``run_inference``.
        graphs: Un-normalised ground-truth graphs (CPU tensors).
        filepath: Path to the source mesh file (stem used in filenames).
        visualizer: ``GraphVisualizer`` bound to the loaded mesh.
        mode: ``"bottom"`` or ``"all"``.
        directory: Directory where HTML plot files are written.
        n_random: Number of random samples to visualize.
        fields: List of field names to include in the plots.
    """
    suffix = "_bottom" if mode == "bottom" else ""

    def _make_paths(tag: str) -> PlotPaths:
        return PlotPaths(
            true=directory / f"{filepath.stem}_{tag}_true{suffix}.html",
            pred=directory / f"{filepath.stem}_{tag}_pred{suffix}.html",
            error=directory / f"{filepath.stem}_{tag}_error{suffix}.html",
        )

    n_random = min(n_random, len(results))
    print(f"Generating {n_random} random plots in {directory}...")
    for i in np.random.choice(len(results), size=n_random, replace=False):
        true = graphs[i].cpu()
        pred = results[i].graph
        if fields == "displacement":
            render_displacement(true, pred, visualizer, _make_paths(f"#{i}"))
        else:
            render_stress(true, pred, visualizer, _make_paths(f"#{i}"))
        print(
            f"Saved plot (sample {i}, {filepath.stem}): "
            f"loss={results[i].loss:.6f}, loss75={results[i].loss75:.6f}."
        )


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
        help="Number of random samples to visualize.",
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
    data = torch.load(args.dataset, weights_only=False)
    graphs = [g.to(device) for g in data["graphs"]]
    msh_path: Path = data["mesh"]
    print(f"Loaded dataset '{args.dataset}' with {len(graphs)} graphs.")
    print(
        f"Each graph has {graphs[0].num_nodes} nodes and {graphs[0].num_edges} edges."
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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

    print("Results:")
    print(f"  Avg L1 loss:              {avg_loss:.6f}")
    print(f"  Avg L1 loss (75th pct):   {avg_loss75:.6f}")

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

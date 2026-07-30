import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import meshio
import numpy as np
import torch
import torch.nn.functional as F

from meshnet.mgn.graphs import GraphVisualizer
from meshnet.mgn.nets import EncodeProcessDecode, MeshGraphNet
from meshnet.mgn.normalizer import GraphNormalizer, LogNormalizer, Normalizer
from meshnet.mgn.utils import msh_to_trimesh


@dataclass
class PredictionResult:
    """Stores the predicted graph and its evaluation metrics for one sample."""

    graph: object
    mae: float  # mean absolute error
    rmae: float  # relative mean absolute error
    pre: float  # peak relative error


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


def evaluate(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float, float]:
    """Compute MAE, RMAE, and PRE between predicted and true values.

    Args:
        pred: Predicted values.
        true: Ground-truth values.

    Returns:
        Tuple of (MAE, RMAE, PRE).
    """
    pred_max = torch.max(torch.abs(pred)).item()
    true_max = torch.max(torch.abs(true)).item()
    mae = F.l1_loss(pred, true).item()
    rmae = mae / true_max
    pre = abs(pred_max - true_max) / true_max

    return mae, rmae, pre


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


def render_von_mises(g_true, g_pred, visualizer: GraphVisualizer, paths: PlotPaths):
    """Render and save ground-truth, prediction, and absolute-error plots.

    Args:
        g_true: Graph carrying ground-truth labels.
        g_pred: Graph carrying predicted labels.
        visualizer: ``GraphVisualizer`` instance bound to the mesh.
        paths: Output file paths for the three plot files.
    """
    # Use true von mises to set the color scale for both plots.
    vm_true = visualizer.compute_von_mises(g_true)
    clim = (vm_true.min().item(), vm_true.max().item())
    for graph, out_path in (
        (g_true, paths.true),
        (g_pred, paths.pred),
    ):
        visualizer.von_mises(graph, clim=clim, save_path=out_path)


def render_displacement(g_true, g_pred, visualizer: GraphVisualizer, paths: PlotPaths):
    """Render and save ground-truth, prediction, and absolute-error plots.

    Args:
        g_true: Graph carrying ground-truth labels.
        g_pred: Graph carrying predicted labels.
        visualizer: ``GraphVisualizer`` instance bound to the mesh.
        paths: Output file paths for the three plot files.
    """
    # Use true displacement to set the color scale for both plots.
    vals = torch.linalg.norm(g_true.y[: g_true.num_physical_nodes, :3], dim=1)
    clim = (vals.min().item(), vals.max().item())

    for graph, out_path in (
        (g_pred, paths.pred),
        (g_true, paths.true),
    ):
        visualizer.displacement(graph, clim=clim, save_path=out_path)


# ---------------------------------------------------------------------------
# Model / normalizer helpers
# ---------------------------------------------------------------------------


def build_normalizer(checkpoint: dict, device: torch.device):
    """Reconstruct the normalizer from a model checkpoint.

    Args:
        checkpoint: Dict loaded from a ``.pth`` checkpoint file.
        device: Target compute device.

    Returns:
        A fitted normalizer instance.
    """
    normalizer = GraphNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    normalizer.fitted = True
    return normalizer.to(device)


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

            mae, rmae, pre = evaluate(y_pred, y_true)

            results.append(
                PredictionResult(graph=g_pred.cpu(), mae=mae, rmae=rmae, pre=pre)
            )
    elapsed = time.time() - start

    return results, elapsed


# ---------------------------------------------------------------------------
# Plot saving
# ---------------------------------------------------------------------------


def save_prediction_visualizations(
    results: Sequence[PredictionResult],
    graphs: Sequence,
    source_path: Path,
    visualizer: GraphVisualizer,
    mode: str,
    output_dir: Path,
    random_sample_count: int = 1,
    field: Literal["displacement", "von_mises"] = "displacement",
) -> None:
    """Save visualizations for the best and worst prediction and randomly selected samples.

    The worst prediction is the sample with the highest relative mean absolute
    error (RMAE). Plot files are written as HTML files in ``output_dir``.

    Args:
        results: Prediction metrics and predicted graphs for each sample.
        graphs: Unnormalized ground-truth graphs corresponding to ``results``.
        source_path: Source mesh path. Its stem is used in output filenames.
        visualizer: Visualizer configured for the source mesh.
        mode: Visualization mode. ``"bottom"`` adds a ``"_bottom"`` suffix.
        output_dir: Directory in which to save the generated HTML files.
        random_sample_count: Maximum number of random samples to visualize.
        field: Physical field to visualize.

    Raises:
        ValueError: If ``results`` is empty or its length differs from ``graphs``.
    """
    filename_suffix = "_b" if mode == "bottom" else ""
    field_suffix = "disp" if field == "displacement" else "vm"

    def build_plot_paths(sample_tag: str) -> PlotPaths:
        filename_prefix = f"{source_path.stem}_{sample_tag}"

        return PlotPaths(
            true=output_dir / f"{filename_prefix}_true{filename_suffix}.html",
            pred=output_dir / f"{filename_prefix}_pred{filename_suffix}.html",
            error=output_dir / f"{filename_prefix}_error{filename_suffix}.html",
        )

    def save_sample(sample_index: int, label: str) -> None:
        result = results[sample_index]
        ground_truth = graphs[sample_index].cpu()
        paths = build_plot_paths(f"{label}_{sample_index}_{field_suffix}")

        if field == "displacement":
            render_displacement(
                ground_truth,
                result.graph,
                visualizer,
                paths,
            )
        else:
            render_von_mises(
                ground_truth,
                result.graph,
                visualizer,
                paths,
            )

        print(
            f"Saved {label} prediction visualization "
            f"(sample={sample_index}, mesh={source_path.stem}, "
            f"mae={result.mae:.6f}, "
            f"rmae={result.rmae:.2%}, "
            f"pre={result.pre:.2%})."
        )

    # Save the sample with the highest relative error.
    worst_sample_index = max(
        range(len(results)),
        key=lambda index: results[index].rmae,
    )
    save_sample(worst_sample_index, label="worst")

    # Save the sample with the lowest relative error.
    best_sample_index = min(
        range(len(results)),
        key=lambda index: results[index].rmae,
    )
    save_sample(best_sample_index, label="best")

    # Save a non-repeating random subset of the remaining samples.
    candidate_indices = [
        index for index in range(len(results)) if index != worst_sample_index
    ]
    sample_count = min(random_sample_count, len(candidate_indices))

    if sample_count == 0:
        return

    print(
        f"Generating {sample_count} random prediction visualizations in {output_dir}."
    )

    random_indices = np.random.choice(
        candidate_indices,
        size=sample_count,
        replace=False,
    )

    for sample_index in random_indices:
        save_sample(int(sample_index), label="random")


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
        default=0,
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

    model = MeshGraphNet(
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
    avg_mae = sum(r.mae for r in results) / n
    avg_rmae = sum(r.rmae for r in results) / n
    avg_pre = sum(r.pre for r in results) / n

    print("Results:")
    print(f"  Avg L1 loss:              {avg_mae:.1f}")
    print(f"  Avg relative L1 loss:     {100 * avg_rmae:.1f}%")
    print(f"  Avg relative peak error:  {100 * avg_pre:.1f}%")

    # Optionally save visualisation plots
    if args.plots:
        visualizer = GraphVisualizer(
            msh_to_trimesh(meshio.read(msh_path)), jupyter_backend=False
        )
        save_prediction_visualizations(
            results,
            graphs,
            msh_path,
            visualizer,
            args.mode,
            args.plot_dir,
            random_sample_count=args.n,
            field=args.target,
        )


if __name__ == "__main__":
    main()

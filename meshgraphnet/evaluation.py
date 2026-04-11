import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from graph_builder import GraphVisualizer
from nets import EncodeProcessDecode
from normalizer import LogNormalizer, Normalizer
from simulator import Simulator
from utils import get_weight, msh_to_trimesh

DATA_FILE = "data/L-Bracket3_100.pt"
CHECKPOINT_FILE = "models/Model0.pth"
TARGET_INDEX = 3


@dataclass
class EvalSummary:
    mae: float
    mae75: float
    node_pct_err: float
    kendall_tau: float
    elapsed_s: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare EncodeProcessDecode and Simulator metrics."
    )
    p.add_argument("--data-file", type=str, default=DATA_FILE)
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE)
    p.add_argument("--target-index", type=int, default=TARGET_INDEX)
    p.add_argument("--mode", choices=["all", "bottom"], default="bottom")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--plots-dir", type=str, default="plots")
    return p.parse_args()


def load_epd_model_and_normalizer(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    params = checkpoint["params"]

    normalizer = build_normalizer(checkpoint, device)
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

    return model, normalizer


def build_normalizer(checkpoint: dict, device: torch.device):
    params = checkpoint["params"]
    stats = checkpoint["stats"]
    kwargs = dict(
        num_features=params["node_dim"],
        num_categorical=params["num_categorical"],
        device=device,
        stats=stats,
    )
    if checkpoint.get("normalizer") == "LogNormalizer":
        return LogNormalizer(**kwargs)
    return Normalizer(**kwargs)


def prepare_graphs(graphs, normalizer, mode):
    normalized_graphs = []

    for graph in graphs:
        graph_norm = normalizer.normalize(graph)
        weight = get_weight(graph.x[:, 2], 1, mode=mode)

        # Only weight physical nodes, not virtual nodes.
        physical = (graph.x[:, -1] != 1.0).unsqueeze(1).float()
        weight = weight * physical

        graph_norm.weight = weight
        graph_norm.y = graph.y
        normalized_graphs.append(graph_norm)

    return normalized_graphs


def _valid_arrays(pred: np.ndarray, true: np.ndarray, weight: np.ndarray):
    pred = np.asarray(pred)
    true = np.asarray(true)
    weight = np.asarray(weight)

    if weight.ndim < pred.ndim:
        repeat = pred.shape[1] if pred.ndim == 2 else 1
        weight = np.repeat(weight, repeat, axis=1)

    mask = weight > 0
    return pred[mask], true[mask]


def mae75(pred: np.ndarray, true: np.ndarray, weight: np.ndarray) -> float:
    x, y = _valid_arrays(pred, true, weight)
    if y.size == 0:
        return float("nan")

    threshold = np.percentile(y, 75)
    top_mask = y >= threshold
    if not np.any(top_mask):
        return float("nan")
    return np.abs(x - y)[top_mask].mean().item()


def compute_kendall_tau(
    pred: np.ndarray, true: np.ndarray, weight: np.ndarray
) -> float:
    x, y = _valid_arrays(pred, true, weight)
    if y.size < 2:
        return float("nan")

    tau = kendalltau(x, y).statistic
    if tau is None or np.isnan(tau):
        return 0.0
    return float(tau)


def aggregate(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def evaluate_encode_process_decode(
    graphs_raw,
    checkpoint_path: str,
    target_index: int,
    mode: str,
    device: torch.device,
) -> EvalSummary:
    model, normalizer = load_epd_model_and_normalizer(checkpoint_path, device)

    graphs_device = [g.to(device) for g in graphs_raw]
    normalized_graphs = prepare_graphs(graphs_device, normalizer, mode)

    maes = []
    maes75 = []
    node_pct_errs = []
    taus = []

    start = time.time()
    with torch.no_grad():
        for g in normalized_graphs:
            y_pred = normalizer.denormalize_y(model(g))[
                :, target_index : target_index + 1
            ]
            y_true = g.y[:, target_index : target_index + 1]

            mae = F.l1_loss(y_pred, y_true, weight=g.weight).item()
            pred_np = y_pred.cpu().numpy()
            true_np = y_true.cpu().numpy()
            weight_np = g.weight.cpu().numpy()

            maes.append(mae)
            maes75.append(mae75(pred_np, true_np, weight_np))
            taus.append(compute_kendall_tau(pred_np, true_np, weight_np))
            node_pct_errs.append(100 * mae / (np.max(true_np * weight_np) + 1e-8))
    elapsed = time.time() - start

    return EvalSummary(
        mae=aggregate(maes),
        mae75=aggregate(maes75),
        kendall_tau=aggregate(taus),
        elapsed_s=elapsed,
        node_pct_err=aggregate(node_pct_errs),
    )


def evaluate_simulator(
    graphs_raw,
    msh_path: str,
    target_index: int,
    mode: str,
    device: torch.device,
) -> EvalSummary:
    if target_index != 3:
        raise ValueError(
            "Simulator comparison currently supports target-index 3 (Von Mises stress) only."
        )

    mesh = meshio.read(msh_path)
    simulator = Simulator(msh_path, std=0.02)

    maes = []
    maes75 = []
    taus = []
    node_pct_errs = []

    start = time.time()
    for g in graphs_raw:
        loads = g.contacts
        uh = simulator.run(loads)
        vm = simulator.compute_vm1(uh)
        y_pred = torch.from_numpy(simulator.probe(vm, mesh.points)).to(device)

        mask = g.x[:, -1] != 1.0
        y_true = g.y[mask, target_index : target_index + 1].to(device)
        weight = get_weight(g.x[mask, 2], y_true.shape[1], mode="bottom")

        mae = F.l1_loss(y_pred, y_true, weight=weight).item()
        pred_np = y_pred.detach().cpu().numpy()
        true_np = y_true.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()

        maes.append(mae)
        maes75.append(mae75(pred_np, true_np, weight_np))
        taus.append(compute_kendall_tau(pred_np, true_np, weight_np))
        node_pct_errs.append(100 * mae / (np.max(true_np * weight_np) + 1e-8))

    elapsed = time.time() - start

    return EvalSummary(
        mae=aggregate(maes),
        mae75=aggregate(maes75),
        kendall_tau=aggregate(taus),
        elapsed_s=elapsed,
        node_pct_err=aggregate(node_pct_errs),
    )


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


def print_comparison(epd: EvalSummary, sim: EvalSummary):
    print("\nComparison (EncodeProcessDecode vs Simulator)")
    print("-" * 60)
    print(f"{'Metric':<18}{'EncodeProcessDecode':>20}{'Simulator':>14}")
    print("-" * 60)
    print(f"{'MAE':<18}{epd.mae:>20.6f}{sim.mae:>14.6f}")
    print(f"{'MAE75':<18}{epd.mae75:>20.6f}{sim.mae75:>14.6f}")
    print(f"{'Node % Error':<18}{epd.node_pct_err:>20.2f}%{sim.node_pct_err:>14.2f}%")
    print(f"{'Kendall Tau':<18}{epd.kendall_tau:>20.4f}{sim.kendall_tau:>14.4f}")
    print(f"{'Runtime (s)':<18}{epd.elapsed_s:>20.2f}{sim.elapsed_s:>14.2f}")
    print("-" * 60)

    mae_delta = pct_change(sim.mae, epd.mae)
    mae75_delta = pct_change(sim.mae75, epd.mae75)
    node_pct_delta = pct_change(sim.node_pct_err, epd.node_pct_err)
    tau_delta = epd.kendall_tau - sim.kendall_tau
    speedup = sim.elapsed_s / epd.elapsed_s if epd.elapsed_s > 0 else float("nan")

    print("Relative difference of EncodeProcessDecode against Simulator:")
    print(f"  MAE:     {mae_delta:+.2f}% (lower is better)")
    print(f"  MAE75:   {mae75_delta:+.2f}% (lower is better)")
    print(f"  Node % Error: {node_pct_delta:+.2f}% (lower is better)")
    print(f"  Tau:     {tau_delta:+.4f} (higher is better)")
    print(f"  Speedup: {speedup:.2f}x faster")


def _run_epd_sample_prediction(
    graph_raw,
    checkpoint_path: str,
    mode: str,
    device: torch.device,
):
    model, normalizer = load_epd_model_and_normalizer(checkpoint_path, device)
    graph = graph_raw.clone().to(device)

    graph_norm = prepare_graphs([graph], normalizer, mode)[0]
    with torch.no_grad():
        y_pred = normalizer.denormalize_y(model(graph_norm)).detach().cpu()

    pred_graph = graph_raw.clone().cpu()
    pred_graph.y = y_pred
    return pred_graph


def _run_simulator_sample_prediction(graph_raw, mesh, msh_path: str):
    simulator = Simulator(msh_path, std=0.02)
    uh = simulator.run(graph_raw.contacts)
    vm = simulator.compute_vm1(uh)
    vm_pred = torch.from_numpy(simulator.probe(vm, mesh.points)).float()

    pred_graph = graph_raw.clone().cpu()
    n_phys = pred_graph.num_physical_nodes
    pred_graph.y[:n_phys, 3] = vm_pred.squeeze(-1)
    return pred_graph


def _stress_clim(mode: str, graph_true, graph_epd, graph_sim):
    n_phys = graph_true.num_physical_nodes
    z = graph_true.x[:n_phys, 2]
    if mode == "bottom":
        mask = torch.isclose(z, torch.zeros_like(z), atol=1e-6)
    else:
        mask = torch.ones_like(z, dtype=torch.bool)

    vals = torch.cat(
        [
            graph_true.y[:n_phys, 3][mask],
            graph_epd.y[:n_phys, 3][mask],
            graph_sim.y[:n_phys, 3][mask],
        ]
    )
    return (vals.min().item(), vals.max().item())


def plot_sample_predictions(
    graphs_raw,
    msh_path: str,
    checkpoint_path: str,
    sample_index: int,
    mode: str,
    device: torch.device,
    plots_dir: str,
):
    if sample_index < 0 or sample_index >= len(graphs_raw):
        raise ValueError(
            f"sample-index={sample_index} is out of range for {len(graphs_raw)} graphs."
        )

    mesh = meshio.read(msh_path)
    visualizer = GraphVisualizer(msh_to_trimesh(mesh), jupyter_backend=False)

    graph_true = graphs_raw[sample_index].clone().cpu()
    graph_epd = _run_epd_sample_prediction(
        graph_true,
        checkpoint_path=checkpoint_path,
        mode=mode,
        device=device,
    )
    graph_sim = _run_simulator_sample_prediction(graph_true, mesh, msh_path)

    clim = _stress_clim(mode, graph_true, graph_epd, graph_sim)
    print(f"\nColor scale limits for stress plots: {clim}")

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(msh_path).stem
    suffix = f"{stem}_sample{sample_index}_{mode}"

    gt_path = out_dir / f"{suffix}_ground_truth.html"
    epd_path = out_dir / f"{suffix}_epd_pred.html"
    sim_path = out_dir / f"{suffix}_simulator_pred.html"

    if mode == "bottom":
        visualizer.bottom(graph_true, clim=clim, save_path=str(gt_path))
        visualizer.bottom(graph_epd, clim=clim, save_path=str(epd_path))
        visualizer.bottom(graph_sim, clim=clim, save_path=str(sim_path))
    else:
        visualizer.stress(graph_true, clim=clim, save_path=str(gt_path))
        visualizer.stress(graph_epd, clim=clim, save_path=str(epd_path))
        visualizer.stress(graph_sim, clim=clim, save_path=str(sim_path))

    print("\nSaved sample plots:")
    print(f"  Ground truth: {gt_path}")
    print(f"  EncodeProcessDecode prediction: {epd_path}")
    print(f"  Simulator prediction: {sim_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    data = torch.load(args.data_file, weights_only=False)
    graphs_raw = data["graphs"]
    msh_path = data["mesh"]

    print(f"Loaded dataset '{args.data_file}' with {len(graphs_raw)} graphs.")
    print(
        f"Using target-index={args.target_index}, mode='{args.mode}', device='{device}'."
    )

    print("\nEvaluating EncodeProcessDecode...")
    epd = evaluate_encode_process_decode(
        graphs_raw=graphs_raw,
        checkpoint_path=args.checkpoint,
        target_index=args.target_index,
        mode=args.mode,
        device=device,
    )

    print("Evaluating Simulator...")
    sim = evaluate_simulator(
        graphs_raw=graphs_raw,
        msh_path=msh_path,
        target_index=args.target_index,
        mode=args.mode,
        device=device,
    )

    print_comparison(epd, sim)

    print("\nGenerating one-sample prediction plots...")
    plot_sample_predictions(
        graphs_raw=graphs_raw,
        msh_path=msh_path,
        checkpoint_path=args.checkpoint,
        sample_index=args.sample_index,
        mode=args.mode,
        device=device,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()

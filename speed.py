import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import gmsh
import meshio
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graph_builder import GraphBuilderVirtual
from nets import EncodeProcessDecode
from normalizer import LogNormalizer, Normalizer
from simulator import Simulator
from utils import get_weight

DATA_FILE = "data/Bushing2_100.pt"
CHECKPOINT_FILE = "models/Model0.pth"
TARGET_INDEX = 3


@dataclass
class SpeedSummary:
    total_s: float
    mean_s: float
    runs: int


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark single-sample runtime.")
    p.add_argument("--data-file", type=str, default=DATA_FILE)
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE)
    p.add_argument("--target-index", type=int, default=TARGET_INDEX)
    p.add_argument("--mode", choices=["all", "bottom"], default="bottom")
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in the dataset to benchmark.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Measured timing iterations per method.",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.02,
        help="Gaussian std for graph builder and simulator.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


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
        weight = get_weight(graph.x[:, 2], 4, mode=mode)

        # Only weight physical nodes, not virtual nodes.
        physical = (graph.x[:, -1] != 1.0).unsqueeze(1).float()
        weight = weight * physical

        graph_norm.weight = weight
        graph_norm.y = graph.y
        normalized_graphs.append(graph_norm)

    return normalized_graphs


def aggregate(values: list[float]) -> SpeedSummary:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return SpeedSummary(
            total_s=float("nan"),
            mean_s=float("nan"),
            runs=0,
        )
    return SpeedSummary(
        total_s=float(np.sum(arr)),
        mean_s=float(np.mean(arr)),
        runs=int(arr.size),
    )


def benchmark_encode_process_decode(
    sample,
    mesh,
    model,
    normalizer,
    mode: str,
    device: torch.device,
    builder,
    runs: int,
):
    start = perf_counter()

    graphs = [
        builder.build(mesh, contacts=sample.contacts).to(device) for _ in range(runs)
    ]
    graphs_norm = [normalizer.normalize(g) for g in graphs]
    loader = DataLoader(graphs_norm, batch_size=4, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            normalizer.denormalize_y(model(batch))

    elapsed = perf_counter() - start

    return SpeedSummary(
        total_s=elapsed,
        mean_s=elapsed / runs,
        runs=runs,
    )


def benchmark_simulator(sample, msh_path: Path, sigma: float, runs: int):
    start = perf_counter()
    mesh = meshio.read(msh_path)

    for i in range(runs):
        simulator = Simulator(str(msh_path), std=sigma)

        uh = simulator.run(sample.contacts)
        vm = simulator.compute_vm1(uh)
        _ = simulator.probe(vm, mesh.points)

    elapsed = perf_counter() - start

    return SpeedSummary(
        total_s=elapsed,
        mean_s=elapsed / runs,
        runs=runs,
    )


def print_comparison(epd: SpeedSummary, sim1: SpeedSummary, sim2: SpeedSummary):
    print("\nSingle-sample speed comparison")
    print("-" * 56)
    print(f"{'Method':<22}{'Total (s)':>12}{'Mean (s)':>12}{'Runs':>10}")
    print("-" * 56)
    print(
        f"{'EncodeProcessDecode':<22}{epd.total_s:>12.6f}{epd.mean_s:>12.6f}{epd.runs:>10}"
    )
    print(
        f"{'Simulator (Coarse)':<22}{sim1.total_s:>12.6f}{sim1.mean_s:>12.6f}{sim1.runs:>10}"
    )
    print(
        f"{'Simulator (Fine)':<22}{sim2.total_s:>12.6f}{sim2.mean_s:>12.6f}{sim2.runs:>10}"
    )
    print("-" * 56)

    speedup = sim1.mean_s / epd.mean_s if epd.mean_s > 0 else float("nan")
    print(f"EncodeProcessDecode speedup over Simulator (Coarse): {speedup:.2f}x")

    speedup = sim2.mean_s / epd.mean_s if epd.mean_s > 0 else float("nan")
    print(f"EncodeProcessDecode speedup over Simulator (Fine): {speedup:.2f}x")


def main():
    args = parse_args()
    device = torch.device(args.device)

    data = torch.load(args.data_file, weights_only=False)
    graphs = data["graphs"]
    if not graphs:
        raise ValueError("Dataset contains no samples.")

    if args.sample_index < 0 or args.sample_index >= len(graphs):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range [0, {len(graphs) - 1}]."
        )

    sample = graphs[args.sample_index]
    msh_path = Path(data["mesh"])
    mesh = meshio.read(msh_path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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

    builder = GraphBuilderVirtual(args.sigma)

    print(f"Loaded dataset: {args.data_file} ({len(graphs)} samples)")
    print(f"Using sample-index={args.sample_index}, mesh={msh_path.name}")
    print(f"Device={device}, runs={args.runs}, sigma={args.sigma}")
    print(f"Graph builder: {builder.__class__.__name__}")

    print("\nBenchmarking EncodeProcessDecode (including graph construction)...")
    epd_speed = benchmark_encode_process_decode(
        sample=sample,
        mesh=mesh,
        model=model,
        normalizer=normalizer,
        mode=args.mode,
        device=device,
        builder=builder,
        runs=args.runs,
    )

    print("Benchmarking Simulator...")
    sim_speed_coarse = benchmark_simulator(
        sample=sample,
        msh_path=msh_path,
        sigma=args.sigma,
        runs=args.runs,
    )

    print("Benchmarking Simulator with finer mesh...")
    msh_path_fine = msh_path.with_name(
        msh_path.stem.replace("cg1", "cg2") + msh_path.suffix
    )
    sim_speed_fine = benchmark_simulator(
        sample=sample,
        msh_path=msh_path_fine,
        sigma=args.sigma,
        runs=args.runs,
    )

    print_comparison(epd_speed, sim_speed_coarse, sim_speed_fine)


if __name__ == "__main__":
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    main()

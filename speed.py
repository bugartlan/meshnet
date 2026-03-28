import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import gmsh
import meshio
import numpy as np
import torch

from graph_builder import GraphBuilderVirtual
from nets import EncodeProcessDecode
from normalizer import LogNormalizer, Normalizer
from simulator import Simulator
from utils import get_weight

DATA_FILE = "data/T-Bracket2_100.pt"
CHECKPOINT_FILE = "models/Train_all_w.pth"
TARGET_INDEX = 3


@dataclass
class SpeedSummary:
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
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
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations not included in timing stats.",
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
        weight = get_weight(graph.x[:, 2], 1, mode=mode)

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
            mean_s=float("nan"),
            std_s=float("nan"),
            min_s=float("nan"),
            max_s=float("nan"),
            runs=0,
        )
    return SpeedSummary(
        mean_s=float(np.mean(arr)),
        std_s=float(np.std(arr)),
        min_s=float(np.min(arr)),
        max_s=float(np.max(arr)),
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
    warmup: int,
    runs: int,
):
    times = []

    # Graph construction is included in this benchmark by rebuilding from mesh + contacts each run.
    for i in range(warmup + runs):
        start = perf_counter()

        y_phys = sample.y[: sample.num_physical_nodes].detach().cpu().numpy()
        rebuilt = builder.build(mesh, y_phys, contacts=sample.contacts)
        g = rebuilt.to(device)

        g_norm = normalizer.normalize(g)
        weight = get_weight(g.x[:, 2], 1, mode=mode)
        weight = weight * (g.x[:, -1] != 1.0).unsqueeze(1).float()
        g_norm.weight = weight

        with torch.no_grad():
            _ = normalizer.denormalize_y(model(g_norm))
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        elapsed = perf_counter() - start
        if i >= warmup:
            times.append(elapsed)

    return aggregate(times)


def benchmark_simulator(
    sample,
    msh_path: Path,
    sigma: float,
    warmup: int,
    runs: int,
):
    times = []
    mesh = meshio.read(msh_path)

    for i in range(warmup + runs):
        start = perf_counter()
        simulator = Simulator(str(msh_path), std=sigma)

        uh = simulator.run(sample.contacts)
        vm = simulator.compute_vm1(uh)
        _ = simulator.probe(vm, mesh.points)

        elapsed = perf_counter() - start
        if i >= warmup:
            times.append(elapsed)

    return aggregate(times)


def print_comparison(epd: SpeedSummary, sim: SpeedSummary):
    print("\nSingle-sample speed comparison")
    print("-" * 84)
    print(
        f"{'Method':<22}{'Mean (s)':>12}{'Std (s)':>12}{'Min (s)':>12}{'Max (s)':>12}{'Runs':>8}"
    )
    print("-" * 84)
    print(
        f"{'EncodeProcessDecode':<22}{epd.mean_s:>12.6f}{epd.std_s:>12.6f}{epd.min_s:>12.6f}{epd.max_s:>12.6f}{epd.runs:>8}"
    )
    print(
        f"{'Simulator':<22}{sim.mean_s:>12.6f}{sim.std_s:>12.6f}{sim.min_s:>12.6f}{sim.max_s:>12.6f}{sim.runs:>8}"
    )
    print("-" * 84)

    speedup = sim.mean_s / epd.mean_s if epd.mean_s > 0 else float("nan")
    print(f"EncodeProcessDecode speedup over Simulator: {speedup:.2f}x")


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
    print(
        f"Device={device}, warmup={args.warmup}, runs={args.runs}, sigma={args.sigma}"
    )
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
        warmup=args.warmup,
        runs=args.runs,
    )

    print("Benchmarking Simulator...")
    sim_speed = benchmark_simulator(
        sample=sample,
        msh_path=msh_path,
        sigma=args.sigma,
        warmup=args.warmup,
        runs=args.runs,
    )

    print_comparison(epd_speed, sim_speed)


if __name__ == "__main__":
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    main()

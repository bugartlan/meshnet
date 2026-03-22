import time

import meshio
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from nets import EncodeProcessDecode, MeshGraphNet
from normalizer import Normalizer
from simulator import Simulator
from utils import get_weight

labels = ["x-displacement", "y-displacement", "z-displacement", "Von Mises Stress"]

DATA_FILE = "data/Test100-2/Bushing3_100.pt"
CHECKPOINT_FILE = "models/Mix250_all_w.pth"
TARGET_INDEX = 3
BATCH_SIZE = 16


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(DATA_FILE, weights_only=False)
    graphs = [g.to(device) for g in data["graphs"]] * 10
    print(f"Loaded dataset '{DATA_FILE}' with {len(graphs)} graphs.")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model_state_dict = checkpoint["model_state_dict"]
    params = checkpoint["params"]

    normalizer = Normalizer(
        num_features=params["node_dim"],
        num_categorical=params["num_categorical"],
        device=device,
        stats=checkpoint["stats"],
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

    # Pre-normalize graph inputs once.
    normalized_graphs = prepare_graphs(graphs, normalizer)
    loader = DataLoader(normalized_graphs, batch_size=BATCH_SIZE, shuffle=False)

    print("Evaluating model on test dataset...")
    total_loss = torch.zeros((), device=device)
    total_nodes = 0
    with torch.no_grad():
        start = time.time()
        for batch in loader:
            y_pred = model(batch)
            y_pred = normalizer.denormalize_y(y_pred)[:, TARGET_INDEX:]
            y_true = batch.y[:, TARGET_INDEX:]

            loss = F.l1_loss(y_pred, y_true, weight=batch.weight)
            num_nodes = batch.weight.sum().item()
            total_loss += loss.detach() * num_nodes
            total_nodes += num_nodes

        end = time.time()

    avg_loss = total_loss / total_nodes
    print(total_loss, total_nodes)
    print(f"Inference completed in {end - start:.2f} seconds.")
    print(f"Average L1 Loss over dataset: {avg_loss:.6f}")

    print("Evaluating FEM (coarse) on test dataset...")
    msh = data["mesh"]
    mesh = meshio.read(msh)
    simulator = Simulator(msh, std=0.01)

    total_loss = 0.0

    start = time.time()
    for g in graphs:
        loads = g.contacts
        uh = simulator.run(loads)
        vm = simulator.compute_vm1(uh)
        y_pred = torch.from_numpy(simulator.probe(vm, mesh.points)).to(device)

        mask = g.x[:, -1] != 1.0
        y_true = g.y[mask, TARGET_INDEX:]
        weight = get_weight(g.x[:, 2], y_true.shape[1], mode="bottom")
        loss = F.l1_loss(y_pred, y_true, weight=weight[mask])
        total_loss += loss.item()
    end = time.time()
    avg_loss = total_loss / len(graphs)
    print(f"FEM evaluation completed in {end - start:.2f} seconds.")
    print(f"Average L1 Loss over dataset: {avg_loss:.6f}")


if __name__ == "__main__":
    main()

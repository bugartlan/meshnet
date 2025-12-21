import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import trimesh
from torch_geometric.loader import DataLoader

from create_dataset import visualize
from nets import EncodeProcessDecode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

graphs = torch.load("cantilever_10.pt", weights_only=False)
loader = DataLoader(graphs, batch_size=1, shuffle=False)

mesh = trimesh.load_mesh("cantilever.stl")
g = graphs[7]
node_dim = g.num_node_features["node"]
mesh_edge_dim = g.num_edge_features["node", "mesh", "node"]
contact_edge_dim = g.num_edge_features["node", "contact", "node"]
latent_dim = 128
output_dim = 1  # predicting Von Mises stress

model = EncodeProcessDecode(
    node_dim=node_dim,
    mesh_edge_dim=mesh_edge_dim,
    contact_edge_dim=contact_edge_dim,
    output_dim=output_dim,
    latent_dim=latent_dim,
    use_layer_norm=False,
).to(device)

checkpoint = torch.load(
    "model.pth", map_location=torch.device("cpu"), weights_only=True
)
model.load_state_dict(checkpoint)

visualize(mesh, g)
g_pred = g.clone()
g_pred["node"].y = model(g_pred.to(device)).detach()
visualize(mesh, g_pred)

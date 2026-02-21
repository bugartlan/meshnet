import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool


def make_mlp(input_dim, hidden_dim, output_dim, layer_norm):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    if layer_norm:
        return nn.Sequential(model, nn.LayerNorm(output_dim))
    else:
        return model


class Processor(nn.Module):
    def __init__(self, dim, layer_norm=False):
        super().__init__()
        self.edge_mlp = make_mlp(3 * dim, dim, dim, layer_norm)
        self.node_mlp = make_mlp(2 * dim, dim, dim, layer_norm)

    def _aggregate_edges(self, edge_index, edge_attr, num_nodes):
        device = edge_attr.device
        dtype = edge_attr.dtype

        D = edge_attr.size(1)
        idx = edge_index[1].unsqueeze(-1).expand(-1, D)
        agg = torch.zeros((num_nodes, D), device=device, dtype=dtype)
        return torch.scatter_add(agg, 0, idx, edge_attr)

    def forward(self, g: Data) -> Data:
        # Update mesh edges
        src, dst = g.edge_index
        edge_cat = torch.cat([g.x[src], g.x[dst], g.edge_attr], dim=-1)
        edge_delta = self.edge_mlp(edge_cat)
        edge_attr = g.edge_attr + edge_delta

        # Update nodes
        agg_edges = self._aggregate_edges(g.edge_index, edge_attr, g.x.size(0))

        # Global context via max pooling
        # batch = (
        #     g.batch
        #     if hasattr(g, "batch") and g.batch is not None
        #     else torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)
        # )
        # global_context = global_max_pool(g.x, batch)  # Shape: [num_graphs, dim]
        # global_expanded = global_context[batch]  # Shape: [num_nodes, dim]

        node_cat = torch.cat([g.x, agg_edges], dim=-1)
        node_delta = self.node_mlp(node_cat)
        x = g.x + node_delta

        return Data(x=x, edge_index=g.edge_index, edge_attr=edge_attr)


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        latent_dim=128,
        message_passing_steps=10,
        use_layer_norm=False,
    ):
        super().__init__()

        self._output_dim = output_dim
        self._latent_dim = latent_dim
        self._message_passing_steps = message_passing_steps
        self._use_layernorm = use_layer_norm

        self._node_dim = node_dim
        self._edge_dim = edge_dim

        self._node_encoder = make_mlp(node_dim, latent_dim, latent_dim, use_layer_norm)
        self._edge_encoder = make_mlp(edge_dim, latent_dim, latent_dim, use_layer_norm)
        self._processor = Processor(latent_dim, layer_norm=use_layer_norm)
        self._decoder = make_mlp(latent_dim, latent_dim, output_dim, False)

    def _encode(self, g: Data):
        x = self._node_encoder(g.x)
        edge_attr = self._edge_encoder(g.edge_attr)

        return Data(x=x, edge_index=g.edge_index, edge_attr=edge_attr)

    def forward(self, g: Data):
        g = self._encode(g)
        for _ in range(self._message_passing_steps):
            g = self._processor(g)
        return self._decoder(g.x)

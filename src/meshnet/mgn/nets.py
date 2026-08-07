import torch
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_mean


def make_mlp(input_dim, hidden_dim, output_dim, layer_norm):
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ]

    if layer_norm:
        layers.append(nn.LayerNorm(output_dim))

    return nn.Sequential(*layers)


class Processor(nn.Module):
    def __init__(self, dim, layer_norm=False):
        super().__init__()
        self.edge_mlp = make_mlp(3 * dim, dim, dim, layer_norm)
        self.node_mlp = make_mlp(2 * dim, dim, dim, layer_norm)

    def forward(self, x, edge_index, edge_attr) -> Data:
        src, dst = edge_index

        # Edge updates
        edge_delta = self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        edge_attr = edge_attr + edge_delta

        # Sum incoming edges
        idx = dst.unsqueeze(-1).expand_as(edge_attr)
        agg = edge_attr.new_zeros(x.size(0), edge_attr.size(-1))
        agg.scatter_add_(0, idx, edge_attr)

        # Node updates
        node_delta = self.node_mlp(torch.cat([x, agg], dim=-1))
        x += node_delta

        return x, edge_attr


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

        self._message_passing_steps = message_passing_steps

        self._node_encoder = make_mlp(node_dim, latent_dim, latent_dim, use_layer_norm)
        self._edge_encoder = make_mlp(edge_dim, latent_dim, latent_dim, use_layer_norm)
        self._processor = Processor(latent_dim, layer_norm=use_layer_norm)
        self._decoder = make_mlp(latent_dim, latent_dim, output_dim, False)

    def forward(self, g: Data):
        x = self._node_encoder(g.x)
        edge_attr = self._edge_encoder(g.edge_attr)
        edge_index = g.edge_index

        for _ in range(self._message_passing_steps):
            x, edge_attr = self._processor(x, edge_index, edge_attr)

        return self._decoder(x)


class MeshGraphNet(EncodeProcessDecode):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        latent_dim=128,
        message_passing_steps=10,
        use_layer_norm=False,
    ):
        super().__init__(
            node_dim,
            edge_dim,
            output_dim,
            latent_dim,
            message_passing_steps,
            use_layer_norm,
        )
        self._processor = nn.ModuleList(
            [
                Processor(latent_dim, layer_norm=use_layer_norm)
                for _ in range(message_passing_steps)
            ]
        )

    def forward(self, g: Data):
        x = self._node_encoder(g.x)
        edge_attr = self._edge_encoder(g.edge_attr)
        edge_index = g.edge_index

        for processor in self._processor:
            x, edge_attr = processor(x, edge_index, edge_attr)

        return self._decoder(x)


class MultiScaleMeshGraphNet(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,
        latent_dim=128,
        fine_pre_message_passing_steps=2,
        fine_post_message_passing_steps=2,
        coarse_message_passing_steps=4,
        use_layer_norm=False,
    ):
        super().__init__()

        self._node_encoder = make_mlp(node_dim, latent_dim, latent_dim, use_layer_norm)

        self._fine_edge_encoder = make_mlp(
            edge_dim, latent_dim, latent_dim, use_layer_norm
        )
        self._coarse_edge_encoder = make_mlp(
            edge_dim, latent_dim, latent_dim, use_layer_norm
        )

        self._fine_pre_processors = nn.ModuleList(
            [
                Processor(latent_dim, layer_norm=use_layer_norm)
                for _ in range(fine_pre_message_passing_steps)
            ]
        )
        self._fine_post_processors = nn.ModuleList(
            [
                Processor(latent_dim, layer_norm=use_layer_norm)
                for _ in range(fine_post_message_passing_steps)
            ]
        )
        self._coarse_processors = nn.ModuleList(
            [
                Processor(latent_dim, layer_norm=use_layer_norm)
                for _ in range(coarse_message_passing_steps)
            ]
        )

        self._up_mlp = make_mlp(2 * latent_dim, latent_dim, latent_dim, use_layer_norm)

        self._decoder = make_mlp(latent_dim, latent_dim, output_dim, False)

        # TODO: fine to coarse mapping
        # TODO: up mlp

    def forward(self, g: Data):
        x = self._node_encoder(g.x)
        e = self._fine_edge_encoder(g.edge_attr)

        # Fine pre-processing
        for processor in self._fine_pre_processors:
            x, e = processor(x, g.edge_index, e)

        # Fine -> coarse
        cx = scatter_mean(x, self.fine_to_coarse, dim=0)
        ce = self._coarse_edge_encoder(g.coarse_edge_attr)

        # Coarse processing
        for processor in self._coarse_processors:
            cx, ce = processor(cx, g.coarse_edge_index, ce)

        # Coarse -> fine
        x += self._up_mlp(torch.cat([x, cx[self.fine_to_coarse]], dim=-1))

        # Fine local refinement
        for processor in self._fine_post_processors:
            x, e = processor(x, g.edge_index, e)

        return self._decoder(x)

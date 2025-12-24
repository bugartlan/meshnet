import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class Mlp(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=128,
        activation=nn.ReLU,
        activate_final=False,
    ):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        ]
        if activate_final:
            layers.append(activation())
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class Processor(nn.Module):
    def __init__(self, dim, layer_norm=False):
        super().__init__()
        # Update mesh edges
        # self.mesh_edge_mlp = self._make_mlp(3 * dim, dim, layer_norm)
        # Update contact edges
        # self.contact_edge_mlp = self._make_mlp(3 * dim, dim, layer_norm)
        self.edge_mlp = self._make_mlp(3 * dim, dim, layer_norm)
        # Update mesh nodes
        self.node_mlp = self._make_mlp(3 * dim, dim, layer_norm)

    def _make_mlp(self, input_dim, output_dim, layer_norm):
        net = Mlp(
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            activate_final=False,
        )

        if layer_norm:
            net = nn.Sequential(net, nn.LayerNorm(output_dim))

        return net

    def _aggregate_edges(self, edge_index, edge_attr, num_nodes):
        device = edge_attr.device
        dtype = edge_attr.dtype

        D = edge_attr.size(1)
        idx = edge_index[1].unsqueeze(-1).expand(-1, D)
        agg = torch.zeros((num_nodes, D), device=device, dtype=dtype)
        deg = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
        deg.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1),
            torch.ones((edge_index.size(1), 1), device=device, dtype=dtype),
        ).clamp_(min=1.0)
        return torch.scatter_add(agg, 0, idx, edge_attr) / deg

    def forward(
        self, x, mesh_edge_index, mesh_edge_attr, contact_edge_index, contact_edge_attr
    ) -> HeteroData:
        # Update mesh edges
        src, dst = mesh_edge_index
        mesh_edge_cat = torch.cat([x[src], x[dst], mesh_edge_attr], dim=-1)
        mesh_edge_delta = self.edge_mlp(mesh_edge_cat)
        new_mesh_edge_attr = mesh_edge_attr + mesh_edge_delta

        # Update contact edges
        src, dst = contact_edge_index
        contact_edge_cat = torch.cat([x[src], x[dst], contact_edge_attr], dim=-1)
        contact_edge_delta = self.edge_mlp(contact_edge_cat)
        new_contact_edge_attr = contact_edge_attr + contact_edge_delta

        # Update nodes
        agg_mesh_edges = self._aggregate_edges(
            mesh_edge_index, new_mesh_edge_attr, x.size(0)
        )
        agg_contact_edges = self._aggregate_edges(
            contact_edge_index, new_contact_edge_attr, x.size(0)
        )
        node_cat = torch.cat([x, agg_mesh_edges, agg_contact_edges], dim=-1)
        node_delta = self.node_mlp(node_cat)
        new_x = x + node_delta

        return new_x, new_mesh_edge_attr, new_contact_edge_attr


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        node_dim,
        mesh_edge_dim,
        contact_edge_dim,
        output_dim,
        latent_dim=128,
        num_layers=2,
        message_passing_steps=10,
        use_layer_norm=False,
    ):
        super().__init__()

        self._output_dim = output_dim
        self._latent_dim = latent_dim
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._use_layernorm = use_layer_norm

        self._node_dim = node_dim
        self._mesh_edge_dim = mesh_edge_dim
        self._contact_edge_dim = contact_edge_dim

        self._node_encoder = self._make_mlp(
            input_dim=self._node_dim,
            output_dim=self._latent_dim,
            layer_norm=self._use_layernorm,
        )
        self._mesh_edge_encoder = self._make_mlp(
            input_dim=self._mesh_edge_dim,
            output_dim=self._latent_dim,
            layer_norm=self._use_layernorm,
        )
        self._contact_edge_encoder = self._make_mlp(
            input_dim=self._contact_edge_dim,
            output_dim=self._latent_dim,
            layer_norm=self._use_layernorm,
        )
        self._processor = Processor(self._latent_dim, layer_norm=self._use_layernorm)
        self._decoder = self._make_mlp(self._latent_dim, self._output_dim, False)

    def _make_mlp(self, input_dim, output_dim, layer_norm):
        net = Mlp(
            input_dim=input_dim,
            hidden_dim=self._latent_dim,
            output_dim=output_dim,
            activate_final=False,
        )

        if layer_norm:
            net = nn.Sequential(net, nn.LayerNorm(output_dim))

        return net

    def _encoder(self, graph: HeteroData):
        # Node features
        new_nodes = self._node_encoder(graph["node"].x)
        # Mesh edges
        new_mesh_edge_index = graph["node", "mesh", "node"].edge_index
        new_mesh_edges = self._mesh_edge_encoder(
            graph["node", "mesh", "node"].edge_attr
        )
        # Contact edges
        new_contact_edge_index = graph["node", "contact", "node"].edge_index
        new_contact_edges = self._contact_edge_encoder(
            graph["node", "contact", "node"].edge_attr
        )

        return (
            new_nodes,
            new_mesh_edge_index,
            new_mesh_edges,
            new_contact_edge_index,
            new_contact_edges,
        )

    def forward(self, graph: HeteroData):
        (
            x,
            mesh_edge_index,
            mesh_edges,
            contact_edge_index,
            contact_edges,
        ) = self._encoder(graph)
        for _ in range(self._message_passing_steps):
            x, mesh_edges, contact_edges = self._processor(
                x,
                mesh_edge_index,
                mesh_edges,
                contact_edge_index,
                contact_edges,
            )
        return self._decoder(x)


class UnsharedEncodeProcessDecode(EncodeProcessDecode):
    def __init__(
        self,
        node_dim,
        mesh_edge_dim,
        contact_edge_dim,
        output_dim,
        latent_dim=128,
        num_layers=2,
        message_passing_steps=10,
        use_layer_norm=False,
    ):
        super().__init__(
            node_dim,
            mesh_edge_dim,
            contact_edge_dim,
            output_dim,
            latent_dim,
            num_layers,
            message_passing_steps,
            use_layer_norm,
        )
        self._processors = nn.ModuleList(
            [Processor(self._latent_dim) for _ in range(self._message_passing_steps)]
        )

    def forward(self, graph: HeteroData):
        (
            x,
            mesh_edge_index,
            mesh_edges,
            contact_edge_index,
            contact_edges,
        ) = self._encoder(graph)
        for _, processor in enumerate(self._processors):
            x, mesh_edges, contact_edges = processor(
                x,
                mesh_edge_index,
                mesh_edges,
                contact_edge_index,
                contact_edges,
            )
        return self._decoder(x)

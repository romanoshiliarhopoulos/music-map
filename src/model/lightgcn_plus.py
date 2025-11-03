import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from typing import Tuple

class LightGNN(nn.Module):
    """
    LightGNN model for collaborative filtering.
    
    This model learns embeddings for playlists and tracks based on the
    bipartite graph structure, without any initial node features.
    """
    
    def __init__(self, num_playlists: int, num_tracks: int, 
                 embedding_dim: int = 64, num_layers: int = 3):
        """
        Args:
            num_playlists: Total number of playlists.
            num_tracks: Total number of tracks.
            embedding_dim: The dimensionality of the embedding space.
            num_layers: The number of GNN layers to stack.
        """
        super().__init__()
        
        self.num_playlists = num_playlists
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_nodes = num_playlists + num_tracks
        
        # Learnable Embeddings
        # We create ONE embedding matrix for all nodes (playlists + tracks).
        self.embedding = nn.Embedding(
            num_embeddings=self.num_nodes,
            embedding_dim=self.embedding_dim
        )
        # Initialize embeddings with a standard distribution
        nn.init.xavier_uniform_(self.embedding.weight)

        # GNN Layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gnn_layers.append(
                pyg_nn.conv.LGConv(normalize=True) 
            )

    def forward(self, n_id: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Performs the forward pass on a *subgraph*.
        
        Args:
            n_id: The global node IDs of the nodes in the subgraph. [Subgraph_Size]
            edge_index: The *local* edge_index of the subgraph. [2, Subgraph_Edges]
            num_nodes: The total number of nodes in this subgraph.
            
        Returns:
            The final embeddings for only the nodes in the subgraph. [Subgraph_Size, D]
        """
        
        # Get initial embeddings for the nodes in the subgraph by looking them up from the full embedding matrix.
        x = self.embedding(n_id) # [Subgraph_Size, D]
        
        all_layer_embeddings = [x]
        current_embeddings = x
        
        for layer in self.gnn_layers:
            # Propagate only on the subgraph.We pass num_nodes to tell the layer the size of the subgraph.
            current_embeddings = layer(current_embeddings, edge_index)
            all_layer_embeddings.append(current_embeddings)
            
        #Final Aggregation (Mean Pooling)
        final_embeddings_stack = torch.stack(all_layer_embeddings, dim=0)
        final_embeddings = torch.mean(final_embeddings_stack, dim=0)
        
        # Return the final embeddings for all nodes in the subgraph
        return final_embeddings

    def predict(self, playlist_embeddings: torch.Tensor, track_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the recommendation score (dot product) between
        a set of playlist embeddings and track embeddings.
        """
        # Simple dot product for collaborative filtering
        return torch.matmul(playlist_embeddings, track_embeddings.T)

    def bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, s_reg_lambda: float = 1e-4) -> torch.Tensor:
        """
        Calculates the Bayesian Personalized Ranking (BPR) loss.
        
        Args:
            pos_scores: Scores for positive (playlist, track) pairs.
            neg_scores: Scores for negative (playlist, track) pairs.
            reg_lambda: L2 regularization strength.
        """
        
        # BPR loss component
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # Simple L2 regularization: regularize the entire embedding matrix

        reg = (self.embedding.weight.norm(2).pow(2) / 2) * s_reg_lambda / self.num_nodes

        return bpr + reg
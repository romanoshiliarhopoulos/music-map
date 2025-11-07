import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from typing import List, Optional, Tuple

class LightGNN(nn.Module):
    """
    LightGNN model for collaborative filtering.
    
    This model learns embeddings for playlists and tracks based on the
    bipartite graph structure, without any node features.
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
        # We create embedding matrix for all nodes (playlists + tracks).
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
        Performs the forward pass on a subgraph.
        
        Args:
            n_id: The global node IDs of the nodes in the subgraph. [Subgraph_Size]
            edge_index: The *local* edge_index of the subgraph. [2, Subgraph_Edges]
            num_nodes: The total number of nodes in this subgraph.
            
        Returns:
            The final embeddings for only the nodes in the subgraph. [Subgraph_Size, Embedding dimension]
        """
        
        # Get initial embeddings for the nodes in the subgraph by looking them up from the full embedding matrix.
        x = self.embedding(n_id) # [Subgraph_Size, D]
        
        all_layer_embeddings = [x]
        current_embeddings = x
        
        for layer in self.gnn_layers:
            # Propagate only on the subgraph. We pass num_nodes to tell the layer the size of the subgraph.
            current_embeddings = layer(current_embeddings, edge_index)
            all_layer_embeddings.append(current_embeddings)
            
        #Final Aggregation (Mean Pooling)
        final_embeddings_stack = torch.stack(all_layer_embeddings, dim=0)
        final_embeddings = torch.mean(final_embeddings_stack, dim=0)
        
        # Return the final embeddings for all nodes in the subgraph
        return final_embeddings

    def predict(self, playlist_id: Optional[int] = None, track_ids: Optional[List[int]] = None, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates top-K track recommendations using the learned embeddings.
        
        This function operates in one of two modes:
        1.  By Playlist ID: If a 'playlist_id' is provided, it finds the
            top-K most similar tracks for that existing playlist. 
            Useful for testing on test (10%) of original data.
        2.  By Track IDs: If a list of 'track_ids' is provided, it
            averages their embeddings to create a temporary "playlist"
            embedding and finds the top-K most similar tracks, excluding
            the ones already in the list.

        Args:
            playlist_id: The global node ID of an existing playlist.
            track_ids: A list of global track IDs for a new/anonymous playlist.
            k: The number of recommendations to return.
            
        Returns:
            A tuple of (top_k_scores, global_track_ids)
        """
        if playlist_id is not None and track_ids is not None:
            raise ValueError("Please provide either 'playlist_id' or 'track_ids', not both.")
            
        if playlist_id is None and track_ids is None:
            raise ValueError("Please provide either 'playlist_id' or 'track_ids'.")

        with torch.no_grad():
            # Get the embeddings, which contain the learned representations
            all_embeddings = self.embedding.weight
            playlist_embeddings = all_embeddings[:self.num_playlists]
            track_embeddings = all_embeddings[self.num_playlists:]
        
        if playlist_id is not None:
            # Mode 1: Recommend for an existing playlist ID
        
            # Get the embedding vector for our query playlist
            query_emb = playlist_embeddings[playlist_id].unsqueeze(0) # Shape: [1, D]
            
            # Calculate scores: [1, D] @ [D, NumTracks] -> [1, NumTracks] using matrix multiply
            scores = torch.matmul(query_emb, track_embeddings.T)
            scores = scores.squeeze() # Shape: [NumTracks]
            
            # Get the top K scores and their local indices
            top_k_scores, top_k_indices = torch.topk(scores, k)
            
            # Convert local track indices to global node IDs
            global_track_ids = top_k_indices + self.num_playlists
            
            return top_k_scores, global_track_ids
            
        elif track_ids is not None:
            # Mode 2: Recommend for a list of tracks
            
            # Convert global track IDs to local indices
            # e.g., [315220, 315221] -> [0, 1]
            try:
                local_track_indices = [tid - self.num_playlists for tid in track_ids]
                local_indices_tensor = torch.tensor(
                    local_track_indices, 
                    dtype=torch.long, 
                    device=all_embeddings.device
                )
            except Exception as e:
                raise ValueError(f"Invalid track_ids provided. Ensure they are global node IDs >= {self.num_playlists}. Error: {e}")

            # Get embeddings for all seed tracks
            seed_track_embs = track_embeddings[local_indices_tensor] 
            
            #Average them to create a new "playlist" embedding
            query_emb = torch.mean(seed_track_embs, dim=0, keepdim=True) # Shape: [1, D]
            
            # Calculate scores against all tracks
            scores = torch.matmul(query_emb, track_embeddings.T)
            scores = scores.squeeze() # Shape: [NumTracks]
            
            # Filter out tracks that were already in the seed list, so they are not recommended
            scores[local_indices_tensor] = -torch.inf
            
            #Get top-K
            top_k_scores, top_k_indices = torch.topk(scores, k)
            
            #Convert local indices to global track IDs
            global_track_ids = top_k_indices + self.num_playlists
            
            return top_k_scores, global_track_ids

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
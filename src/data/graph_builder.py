import torch
import torch_geometric as pyg
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class GraphBuilder:
    """
    Build bipartite playlist-track interaction graph for Spotify MPD
    """
    
    def __init__(self, num_playlists: int, num_tracks: int):
        self.num_playlists = num_playlists
        self.num_tracks = num_tracks
        self.total_nodes = num_playlists + num_tracks
        
    def build_bipartite_graph(self, df: pd.DataFrame, 
                             audio_embeddings: np.ndarray = None) -> Data:
        """
        Build bipartite playlist-track graph
        
        Graph structure:
        - Playlist nodes: [0, num_playlists)
        - Track nodes: [num_playlists, num_playlists + num_tracks)
        - Edges: Bidirectional playlist <-> track connections
        - Edge weights: Interaction weights from preprocessing
        """
        print("Building bipartite playlist-track graph...")
        
        # Create edge indices
        playlist_indices = df['playlist_idx'].values
        track_indices = df['track_idx'].values + self.num_playlists
        
        # Create bidirectional edges
        edge_index = torch.tensor([
            np.concatenate([playlist_indices, track_indices]),
            np.concatenate([track_indices, playlist_indices])
        ], dtype=torch.long)
        
        # Edge weights
        edge_weights = torch.tensor(
            np.concatenate([df['interaction_weight'].values] * 2),
            dtype=torch.float
        )
                
        # Create PyG Data object
        graph_data = Data(
            edge_index=edge_index,
            edge_weight=edge_weights,
            num_nodes=self.total_nodes
        )
        
        print(f"Graph built:")
        print(f"  Nodes: {graph_data.num_nodes:,} ({self.num_playlists:,} playlists + {self.num_tracks:,} tracks)")
        print(f"  Edges: {graph_data.edge_index.shape[1]:,}")
        
        return graph_data
    

import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import pickle
import os
import argparse
import numpy as np

def analyze_graph(args):
    """
    Loads the processed graph and mapping files to print statistics.
    """
    print(f"Analyzing data from: {args.processed_dir}")

    graph_path = os.path.join(args.processed_dir, 'train_graph.pt')
    playlist_map_path = os.path.join(args.processed_dir, 'playlist_mapping.pkl')
    track_map_path = os.path.join(args.processed_dir, 'track_mapping.pkl')

    try:
        with open(playlist_map_path, 'rb') as f:
            playlist_mapping = pickle.load(f)
        with open(track_map_path, 'rb') as f:
            track_mapping = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping files not found in {args.processed_dir}.")
        print("Please run the preprocessing script first.")
        return

    num_playlists = len(playlist_mapping)
    num_tracks = len(track_mapping)

    try:
        data = torch.load(graph_path, weights_only=False)
    except FileNotFoundError:
        print(f"Error: {graph_path} not found.")
        print("Please run 'build_graph.py' first.")
        return

    print("\n--- Graph Statistics ---")
    print(f"  Total Nodes: {data.num_nodes:,}")
    print(f"    - Playlists: {num_playlists:,}")
    print(f"    - Tracks: {num_tracks:,}")
    print(f"  Total Edges (bidirectional): {data.num_edges:,}")

    # Calculate Bipartite Sparsity 
    # We divide by 2 because the graph is bidirectional
    actual_interactions = data.num_edges / 2
    max_possible_interactions = num_playlists * num_tracks
    
    # Sparsity = (actual edges) / (max possible edges)
    bipartite_sparsity = actual_interactions / max_possible_interactions
    
    print(f"  Actual Interactions (unidirectional): {actual_interactions:,.0f}")
    print(f"  Max Possible Interactions: {max_possible_interactions:,}")
    print(f"  Bipartite Sparsity: {bipartite_sparsity:.8f} (or {bipartite_sparsity*100:.6f} %)")

    # Calculate Degree Statistics
    print("\n--- Node Degree Statistics ---")
    # Calculate degree for all nodes
    degree = pyg_utils.degree(data.edge_index[0], num_nodes=data.num_nodes).float()
    
    print(f"  Degree Mean: {degree.mean():.4f}")
    print(f"  Degree Std: {degree.std():.4f}")
    print(f"  Degree Min: {degree.min():.0f}")
    print(f"  Degree Max: {degree.max():.0f}")

    # Separate degrees for playlists and tracks
    # Playlist nodes are in range [0, num_playlists)
    playlist_degrees = degree[:num_playlists]
    # Track nodes are in range [num_playlists, num_playlists + num_tracks)
    track_degrees = degree[num_playlists:]

    print("\n  Playlist Node Degree:")
    print(f"    - Mean: {playlist_degrees.mean():.4f}")
    print(f"    - Min: {playlist_degrees.min():.0f}")
    print(f"    - Max: {playlist_degrees.max():.0f}")
    
    print("\n  Track Node Degree:")
    print(f"    - Mean: {track_degrees.mean():.4f}")
    print(f"    - Min: {track_degrees.min():.0f}")
    print(f"    - Max: {track_degrees.max():.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the processed training graph.")
    
    parser.add_argument(
        '--processed_dir', 
        type=str, 
        default='data/processed',
        help="Directory containing preprocessed data files (train_graph.pt, pkl). (Default: data/processed)"
    )
    
    args = parser.parse_args()
    analyze_graph(args)
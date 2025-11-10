import argparse
import os
import pickle
from src.data.graph_builder import GraphBuilder
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import pickle
import os
import argparse
from typing import Tuple, Dict

def main(args):
    """
    Main function to load data, build the graph, and save it.
    """
    print(f"Loading data from: {args.processed_dir}")
    
    # Define file paths 
    train_path = os.path.join(args.processed_dir, 'train.parquet')
    playlist_map_path = os.path.join(args.processed_dir, 'playlist_mapping.pkl')
    track_map_path = os.path.join(args.processed_dir, 'track_mapping.pkl')
    output_graph_path = os.path.join(args.processed_dir, 'train_graph.pt')
    

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
    
    if num_playlists == 0 or num_tracks == 0:
        print("Error: Mappings are empty. Cannot build graph.")
        return
        
    # Load training data
    print(f"Loading training interactions from {train_path}...")
    try:
        train_df = pd.read_parquet(train_path)
    except FileNotFoundError:
        print(f"Error: {train_path} not found.")
        print("Please run the preprocessing script first.")
        return
        
    print(f"Loaded {len(train_df):,} training interactions.")
    
    # Initialize GraphBuilder 
    builder = GraphBuilder(num_playlists=num_playlists, num_tracks=num_tracks)
    
    # Build the graph 
    graph_data = builder.build_bipartite_graph(train_df)
    
    # Save the graph object 
    print(f"\nSaving graph to {output_graph_path}...")
    torch.save(graph_data, output_graph_path)
    print("Graph saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the training graph from preprocessed data.")
    
    parser.add_argument(
        '--processed_dir', 
        type=str, 
        default='data/processed',
        help="Directory containing preprocessed data files (parquet, pkl). (Default: data/processed)"
    )
    
    args = parser.parse_args()
    main(args)
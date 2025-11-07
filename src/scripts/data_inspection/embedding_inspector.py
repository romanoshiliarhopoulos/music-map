import torch
import argparse
import sys
import json
import pickle
import os
from torch_geometric.data import Data
from typing import Dict, Any, Optional

from src.model.lightgcn_plus import LightGNN


import pandas as pd



# We need these to initialize the model structure before loading the weights
NUM_PLAYLISTS = 315_220
NUM_TRACKS = 176_768
NUM_NODES = NUM_PLAYLISTS + NUM_TRACKS
EMBEDDING_DIM = 32
NUM_LAYERS = 2


def load_model(checkpoint_path: str) -> LightGNN:
    """Initializes the model and loads weights if a checkpoint is provided."""
    
    model = LightGNN(
        num_playlists=NUM_PLAYLISTS,
        num_tracks=NUM_TRACKS,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS
    )
    
    if checkpoint_path:
        try:
            # Load weights onto CPU first
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Successfully loaded TRAINED model weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Using UNTRAINED (randomly initialized) model weights.")
        
    model.eval() # Set model to evaluation mode 
    return model

def get_all_embeddings(model: LightGNN, graph_path: str, device: torch.device) -> torch.Tensor:
    """
    Retrieves the trained (L0) embeddings from the model.
    
    For LightGCN, the 'final' embeddings for inference are typically
    taken directly from the trained embedding layer (model.embedding.weight),
    as the GNN's forward pass is too memory-intensive to run on the full graph.
    The training process bakes the graph structure into these L0 embeddings.
    """
        
    print("Extracting embeddings directly from the model...")
    
    # Move model to the target device to extract weights
    model = model.to(device)
    
    with torch.no_grad():
        final_embeddings = model.embedding.weight.detach().cpu()

    print("Embeddings extracted.")
    return final_embeddings

def load_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads track details (name, artist) from a .parquet file using the global ID."""
    if not filepath:
        print("Warning: No --metadata_file file provided. Will not be able to show track names.", file=sys.stderr)
        return None
    
    try:
        _, ext = os.path.splitext(filepath)
        if ext != '.parquet':
            print(f"Warning: --metadata_file file is not a .parquet file. Skipping.", file=sys.stderr)
            return None
            
        print(f"Loading metadata from {filepath}...")
        
        id_col = 'track_idx' 

        detail_col_names = ['track_name', 'artist_name', 'track_uri']
        
        df = pd.read_parquet(filepath, engine='pyarrow')
        
        # --- Check if required columns exist ---
        if id_col not in df.columns:
            print(f"Error: The parquet file {filepath} must contain a '{id_col}' column.", file=sys.stderr)
            print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
            return None
            
        available_detail_cols = [col for col in detail_col_names if col in df.columns]
        if not available_detail_cols:
             print(f"Error: Parquet file must have at least one of {detail_col_names}", file=sys.stderr)
             return None
        
        all_needed_cols = [id_col] + available_detail_cols
        
        # De-duplicate, keeping only the first entry for each local track_idx
        df = df[all_needed_cols].drop_duplicates(subset=[id_col])
        
        # This maps e.g., track_idx 0 -> global_id 315220
        print(f"Creating global node ID from '{id_col}' + {NUM_PLAYLISTS}...")
        
        # Use the global constant NUM_PLAYLISTS defined at the top of the script
        df['global_node_id'] = df[id_col] + NUM_PLAYLISTS 
        
        # Convert the NEW global_node_id column to string for dict lookup
        df['global_node_id'] = df['global_node_id'].astype(str)
        
        metadata = df.set_index('global_node_id').to_dict('index')
        
        print(f"Successfully loaded details for {len(metadata)} unique tracks.")
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not load/process metadata from {filepath}. Error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LightGNN embeddings and find nearest neighbors."
    )
    parser.add_argument(
        "graph_file", 
        type=str, 
        help="Path to the full 'train_graph.pt' file. (Used for context, but not loaded for this method)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to a model checkpoint (.pt state_dict) to load. (Optional: uses untrained model if not provided)"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="data/processed/train.parquet",
        help="Path to a .parquet file (e.g., train.parquet) with global track IDs, 'track_name', etc."
    )
    parser.add_argument(
        "--playlist_id",
        type=int,
        required=True,
        help="The global node ID of the playlist to inspect (e.g., 123)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest tracks to find."
    )
    args = parser.parse_args()
    
    if args.playlist_id >= NUM_PLAYLISTS:
        print(f"Error: Playlist ID must be less than {NUM_PLAYLISTS}.", file=sys.stderr)
        sys.exit(1)

    device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model (Trained or Untrained)
    model = load_model(args.checkpoint)
    
    # 2. Load Metadata
    # This function now loads the .parquet file and maps by global ID
    metadata = load_metadata(args.metadata_file)
    
    # 3. Get Final Embeddings
    # This now just extracts model.embedding.weight, avoiding the GNN pass
    all_embeddings = get_all_embeddings(model, args.graph_file, device)
    
    # 4. Separate playlist and track embeddings
    # The embeddings are already on the CPU from get_all_embeddings
    playlist_embeddings = all_embeddings[:NUM_PLAYLISTS]
    track_embeddings = all_embeddings[NUM_PLAYLISTS:]
    
    # 5. Find Nearest Neighbors
    print(f"\n--- Finding Top {args.k} Tracks for Playlist {args.playlist_id} ---")
    
    # Get the embedding vector for our query playlist
    query_playlist_emb = playlist_embeddings[args.playlist_id].unsqueeze(0) # [1, D]
    
    # Calculate scores (dot product) against ALL tracks
    # [1, D] @ [D, NumTracks] -> [1, NumTracks]
    scores = query_playlist_emb @ track_embeddings.T
    scores = scores.squeeze() # [NumTracks]
    
    # Get the top K scores and their indices
    top_k_scores, top_k_indices = torch.topk(scores, args.k)
    
    # 6. Print Results
    # The 'top_k_indices' are LOCAL to the track_embeddings tensor.
    # We must add NUM_PLAYLISTS to get their GLOBAL node IDs.
    global_track_ids = top_k_indices + NUM_PLAYLISTS
    
    print("Displaying: Rank | Global Track ID | Track Info | Score")
    for i in range(args.k):
        track_id = global_track_ids[i].item()
        score = top_k_scores[i].item()
        
        # --- Simplified single-step lookup ---
        track_info = "???"
        
        if metadata:
            # Direct lookup using the string of the global track ID
            details = metadata.get(str(track_id))
            
            if details:
                track_name = details.get('track_name', 'N/A')
                artist_name = details.get('artist_name', 'N/A')
                track_info = f"{track_name} - {artist_name}"
            else:
                # Fallback if the ID isn't in the parquet file
                track_info = "Unknown Track ID"
        else:
            track_info = "No metadata provided"
            
        print(f"  Rank {i+1:02d} | {track_id} | {track_info} (Score: {score:.4f})")

if __name__ == "__main__":
    main()


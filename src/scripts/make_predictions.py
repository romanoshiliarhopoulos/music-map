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



NUM_PLAYLISTS = 315_220
NUM_TRACKS = 176_768
NUM_NODES = NUM_PLAYLISTS + NUM_TRACKS
EMBEDDING_DIM = 32
NUM_LAYERS = 2


def load_model(checkpoint_path: str) -> LightGNN:
    """Initializes the model structure and loads checkpoint weights."""
    
    print(f"Initializing model structure: EMBEDDING_DIM={EMBEDDING_DIM}, NUM_LAYERS={NUM_LAYERS}")
    model = LightGNN(
        num_playlists=NUM_PLAYLISTS,
        num_tracks=NUM_TRACKS,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # Load weights onto CPU first
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Successfully loaded TRAINED model weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}", file=sys.stderr)
            sys.exit(1)
    elif checkpoint_path:
         print(f"Warning: Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
         print("Using UNTRAINED (randomly initialized) model weights.")
    else:
        print("Using UNTRAINED (randomly initialized) model weights.")
        
    model.eval() # Set model to evaluation mode 
    return model

def load_json_metadata(filepath: str) -> Optional[Dict[str, str]]:
    """Loads track details (name - artist) from a .json file."""
    
    if not filepath:
        print("Warning: No --metadata_file file provided. Will not be able to show track names.", file=sys.stderr)
        return None
    
    if not os.path.exists(filepath):
        print(f"Warning: Metadata file not found: {filepath}. Skipping.", file=sys.stderr)
        return None

    try:
        _, ext = os.path.splitext(filepath)
        if ext != '.json':
            print(f"Warning: --metadata_file file '{filepath}' is not a .json file. Skipping.", file=sys.stderr)
            return None
            
        print(f"Loading metadata from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        print(f"Successfully loaded details for {len(metadata)} unique tracks.")
        return metadata
        
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON from {filepath}. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Could not load/process metadata from {filepath}. Error: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Get track recommendations from a trained LightGNN model."
    )
    
    #Arguments
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="data/train/checkpoints/model_epoch_100.pt", 
        help="Path to a model checkpoint (.pt state_dict) to load."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="data/processed/track_metadata_mapping.json",
        help="Path to a .parquet file (e.g., train.parquet) with global track IDs, 'track_name', etc."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest tracks to find."
    )

    parser.add_argument(
        "--playlist_id",
        type=int,
        default=None,
        help="The global node ID of the playlist to get recommendations for (e.g., 516)."
    )
    parser.add_argument(
        "--track_ids",
        type=int,
        default=None, 
        nargs='+',
        help="A list of global track IDs to use as a 'seed' for a new playlist (e.g., 443074 319783)."
    )

    args = parser.parse_args()
    
    if args.playlist_id is not None and args.playlist_id >= NUM_PLAYLISTS:
        print(f"Error: Playlist ID must be < {NUM_PLAYLISTS}. Got {args.playlist_id}.", file=sys.stderr)
        sys.exit(1)
        
    if args.track_ids is not None:
        for tid in args.track_ids:
            if tid < NUM_PLAYLISTS:
                print(f"Error: Track ID {tid} is invalid. Track IDs must be >= {NUM_PLAYLISTS}.", file=sys.stderr)
                sys.exit(1)

    # Use MPS if available 
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = load_model(args.checkpoint)

    # Load Metadata
    metadata = load_json_metadata(args.metadata_file)
    
    model = model.to(device)
    
    # Get Recommendations
    top_k_scores = None
    global_track_ids = None

    if args.playlist_id is not None:
        print(f"\n--- Finding Top {args.k} Tracks for Playlist {args.playlist_id} ---")
        try:
            top_k_scores, global_track_ids = model.predict(
                playlist_id=args.playlist_id, 
                k=args.k
            )
        except Exception as e:
            print(f"Error during prediction for playlist {args.playlist_id}: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.track_ids is not None:
        print(f"\n--- Finding Top {args.k} Tracks for new playlist with {len(args.track_ids)} seed tracks ---")
        for tid in args.track_ids:
                track_info = metadata.get(str(tid), "Unknown Track")
                print(f"  Seed Track: {track_info} (ID: {tid})")
        try:
            top_k_scores, global_track_ids = model.predict(
                track_ids=args.track_ids, 
                k=args.k
            )
        except Exception as e:
            print(f"Error during prediction for track IDs {args.track_ids}: {e}", file=sys.stderr)
            sys.exit(1)

    # Print Results
    print("\nDisplaying: Rank | Global Track ID | Track Info | Score")
    for i in range(args.k):

        track_id = global_track_ids[i].cpu().item()
        score = top_k_scores[i].cpu().item()
        
        # --- Metadata Lookup ---
        track_info = "???"
        
        if metadata:
            # Direct lookup using the string of the global track ID
            details = metadata.get(str(track_id))
            
            if details:
                track_info = details
            else:
                # Fallback if the ID isn't in the parquet file
                track_info = f"Track ID {track_id} not in metadata"
        else:
            track_info = "No metadata provided"
            
        print(f"  Rank {i+1:02d} | {track_id} | {track_info} (Score: {score:.4f})")

if __name__ == "__main__":
    main()
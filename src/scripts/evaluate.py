import torch
import pandas as pd
import argparse
import sys
import os
from tqdm import tqdm
from typing import Dict, Any, Optional

from src.model.lightgcn_plus import LightGNN

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
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Successfully loaded TRAINED model weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
         print(f"Warning: Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
         print("Using UNTRAINED (randomly initialized) model weights.")
        
    model.eval() 
    return model


def load_test_data(parquet_file: str) -> pd.Series:
    """
    Loads the test parquet file and groups it by playlist.
    
    Reads'playlist_id' and 'track_idx', then convert'track_idx' to a global node ID.
    """
    try:
        df = pd.read_parquet(parquet_file, columns=['playlist_id', 'track_idx'])
        print(f"Loaded {len(df)} interactions from {parquet_file}")
    
    except KeyError:
        print(f"Error: Could not find 'playlist_id' or 'track_idx' in {parquet_file}.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {parquet_file}. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert local track_idx to global node ID
    df['track_global_id'] = df['track_idx'] + NUM_PLAYLISTS

    # Group by the correct playlist column ('playlist_id') and use the new 'track_global_id' column
    print("Grouping tracks by playlist...")
    playlist_groups = df.groupby('playlist_id')['track_global_id'].apply(lambda x: list(set(x)))
    
    # Filter out playlists that are too short to split
    min_len = 2
    original_count = len(playlist_groups)
    playlist_groups = playlist_groups[playlist_groups.apply(len) >= min_len]
    filtered_count = len(playlist_groups)
    
    if filtered_count == 0:
        print(f"Warning: No playlists found with >= {min_len} tracks. Evaluation cannot proceed.")
    else:
        print(f"Filtered {original_count - filtered_count} playlists with < {min_len} tracks. Testing on {filtered_count} playlists.")
    
    return playlist_groups


def run_evaluation(model: LightGNN, test_playlists: pd.Series, device: torch.device, k: int):
    """
    Runs the cold-start evaluation
    """
    model = model.to(device)
    
    all_recalls = []
    all_precisions = []
    all_hit_rates = []

    print(f"\n--- Starting Evaluation (K={k}) ---")
    
    for pid, all_tracks in tqdm(test_playlists.items(), desc="Evaluating Playlists"):
        
        # Split the playlist (your idea)
        split_point = len(all_tracks) // 2
        
        # Use first half as seed
        seed_tracks = all_tracks[:split_point]
        
        # Use second half as ground truth - set for fast lookup
        ground_truth_tracks = set(all_tracks[split_point:])

        if not seed_tracks or not ground_truth_tracks:
            continue 

        try:
            # Get Top-K Recommendations
            _, top_k_track_ids_tensor = model.predict(
                track_ids=seed_tracks, 
                k=k
            )
            
            # Convert tensor to a set of item IDs
            recommended_tracks = set(top_k_track_ids_tensor.cpu().numpy())

            # Calculate Hits
            hits = ground_truth_tracks.intersection(recommended_tracks)
            num_hits = len(hits)

            #Calculate Metrics
            recall = num_hits / len(ground_truth_tracks)
            precision = num_hits / k
            hit_rate = 1.0 if num_hits > 0 else 0.0
            
            all_recalls.append(recall)
            all_precisions.append(precision)
            all_hit_rates.append(hit_rate)

        except Exception as e:
            print(f"Warning: Failed to evaluate playlist {pid}. Error: {e}", file=sys.stderr)
            continue

    #Aggregate Results
    avg_recall = (sum(all_recalls) / len(all_recalls)) * 100
    avg_precision = (sum(all_precisions) / len(all_precisions)) * 100
    avg_hit_rate = (sum(all_hit_rates) / len(all_hit_rates)) * 100

    print("\n--- 📊 Evaluation Results ---")
    print(f"  Playlists Tested: {len(all_recalls)}")
    print(f"  Average Recall@{k}:    {avg_recall:.2f}%")
    print(f"  Average Precision@{k}:  {avg_precision:.2f}%")
    print(f"  Average Hit Rate@{k}:   {avg_hit_rate:.2f}%")
    print("------------------------------")
    print(f"Recall @{k}: Out of the songs the user actually had, what % did we recommend?")
    print(f"Precision @{k}: Out of our {k} recommendations, what % were correct?")
    print(f"Hit Rate @{k}: What % of the time did we recommend at least ONE correct song?")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LightGNN model.")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="data/train/checkpoints/model_epoch_5000.pt", 
        help="Path to a model checkpoint (.pt state_dict) to load."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test.parquet",
        help="Path to the test.parquet file."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of recommendations to evaluate (K)."
    )

    args = parser.parse_args()
    
    # Use MPS if available 
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load Model
    model = load_model(args.checkpoint)
    
    # Load Test Data
    test_playlists = load_test_data(args.test_file)

    # Run Evaluation
    if not test_playlists.empty:
        run_evaluation(model, test_playlists, device, args.k)
    else:
        print("No valid playlists to evaluate. Exiting.")


if __name__ == "__main__":
    main()
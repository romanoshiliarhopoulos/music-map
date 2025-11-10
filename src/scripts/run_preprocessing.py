import argparse
import time
import os
import glob

from src.data.preprocessor import Preprocessor

def main(args):
    """
    Main function to run the data preprocessing pipeline.
    """
    print("Starting batched preprocessing pipeline...")
    start_time = time.time()
    
    # Define temp directory for batches
    temp_dir = os.path.join(args.output_dir, 'temp_batches')

    # Initialize Preprocessor
    preprocessor = Preprocessor(
        position_weight_decay=args.pos_decay,
        min_track_occurrences=args.min_tracks,
        min_playlist_tracks=args.min_playlist_len
    )
    
    # Get list of all slice files
    all_slice_files = sorted([
        f for f in os.listdir(args.data_path) 
        if f.startswith('mpd.slice.') and f.endswith('.json')
    ])
    
    if not all_slice_files:
        print(f"Error: No 'mpd.slice.*.json' files found in {args.data_path}")
        return
        
    print(f"Found {len(all_slice_files)} total slice files.")

    # Pass 1: Pre-computation 
    preprocessor.precompute_stats(args.data_path, all_slice_files)
    
    # Pass 2: Batch Processing 
    preprocessor.process_and_save_batches(
        data_path=args.data_path,
        slice_files=all_slice_files,
        batch_size=args.batch_size,
        temp_output_dir=temp_dir
    )
    
    # Pass 3: Final Consolidation 
    preprocessor.finalize_processing(
        temp_dir=temp_dir,
        output_dir=args.output_dir
    )

    end_time = time.time()
    print(f"\nPreprocessing pipeline finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Spotify MPD preprocessing pipeline.")
    
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the directory containing 'mpd.slice.*.json' files.")
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help="Directory to save the processed data. (Default: data/processed)")
    

    parser.add_argument('--batch_size', type=int, default=20,
                        help="Number of slice files to process in each batch. (Default: 20)")
    

    parser.add_argument('--min_tracks', type=int, default=10,
                        help="Minimum track occurrences. (Default: 10)")
    parser.add_argument('--min_playlist_len', type=int, default=10,
                        help="Minimum tracks per playlist. (Default: 10)")
    parser.add_argument('--pos_decay', type=float, default=0.001,
                        help="Position weight decay. (Default: 0.001)")
        
    args = parser.parse_args()
    
    main(args)
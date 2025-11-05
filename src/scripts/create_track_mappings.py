import pandas as pd
import pyarrow.parquet as pq
import os
import csv
import json
import sys
from typing import Set, Dict

# --- Configuration ---

NUM_PLAYLISTS = 315_220

DATA_DIR = "data/processed"

INPUT_FILES = [
    os.path.join(DATA_DIR, "train.parquet"),
    os.path.join(DATA_DIR, "test.parquet"),
    os.path.join(DATA_DIR, "val.parquet")
]

# The final output CSV file
OUTPUT_CSV = "track_metadata_mapping.csv"
# The final output JSON file
OUTPUT_JSON = "track_metadata_mapping.json"


COLUMNS_TO_READ = ['track_idx', 'track_name', 'artist_name']

CSV_HEADER = ['track_name', 'track_artist', 'global_track_id']



def create_track_mapping():
    """
    Scans parquet files in chunks to create a de-duplicated CSV map
    and a JSON dictionary of all tracks, using their global node IDs.
    """
    
    seen_local_track_ids: Set[int] = set()
    
    # { "global_id_str": "Track Name - Artist Name" }
    track_map_dict: Dict[str, str] = {}
    
    records_written = 0
    
    print(f"Starting to create track mapping...")
    print(f"Using NUM_PLAYLISTS offset: {NUM_PLAYLISTS}")
    print(f"CSV Output: {OUTPUT_CSV}")
    print(f"JSON Output: {OUTPUT_JSON}")

    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow(CSV_HEADER)
            
            for file_path in INPUT_FILES:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found, skipping: {file_path}", file=sys.stderr)
                    continue
                    
                print(f"\nProcessing file: {file_path}...")
                
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    
                    schema_cols = parquet_file.schema.names
                    if not all(col in schema_cols for col in COLUMNS_TO_READ):
                        print(f"  Warning: File {file_path} is missing one or more required columns {COLUMNS_TO_READ}. Skipping.", file=sys.stderr)
                        continue

                    # Read the file in chunks (batches)
                    for batch in parquet_file.iter_batches(columns=COLUMNS_TO_READ):
                        df_batch = batch.to_pandas()
                        
                        for row in df_batch.itertuples(index=False):
                            local_track_id = row.track_idx
                            
                            # Check if we have already saved this track
                            if local_track_id not in seen_local_track_ids:
                                
                                seen_local_track_ids.add(local_track_id)
                                
                                global_track_id = local_track_id + NUM_PLAYLISTS
                                
                                track_name = str(row.track_name)
                                artist_name = str(row.artist_name)
                                json_value = f"{track_name} - {artist_name}"
                                
                                writer.writerow([track_name, artist_name, global_track_id])
                                
                                # Add to the Python dictionary for JSON
                                track_map_dict[str(global_track_id)] = json_value
                                
                                records_written += 1
                        
                        print(f"  ... processed batch. Total unique tracks found: {records_written}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
                    
    except IOError as e:
        print(f"Error: Could not open or write to output file {OUTPUT_CSV}. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFinished reading all parquet files. Found {records_written} unique tracks.")

    # --- Save the JSON file ---
    print(f"\nSaving JSON dictionary to {OUTPUT_JSON}...")
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as jf:
            json.dump(track_map_dict, jf, indent=4)
        print(f"Successfully saved JSON map.")
    except IOError as e:
        print(f"Error: Could not write to JSON file {OUTPUT_JSON}. {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during JSON serialization: {e}", file=sys.stderr)

    print(f"\n---" * 10)
    print(f"Successfully finished.")
    print(f"Total unique tracks written: {records_written}")
    print(f"CSV map file created at: {OUTPUT_CSV}")
    print(f"JSON map file created at: {OUTPUT_JSON}")


if __name__ == "__main__":
    create_track_mapping()
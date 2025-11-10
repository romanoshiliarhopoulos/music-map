import pandas as pd
import numpy as np
import json
import os
import pickle
import shutil 
from typing import Dict, Tuple, List
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import Counter 

class Preprocessor:
    def __init__(self, position_weight_decay: float = 0.001, 
                 min_track_occurrences: int = 5,
                 min_playlist_tracks: int = 10):
        
        self.position_weight_decay = position_weight_decay
        self.min_track_occurrences = min_track_occurrences
        self.min_playlist_tracks = min_playlist_tracks
        
        self.playlist_encoder = LabelEncoder()
        self.track_encoder = LabelEncoder()
        
        self.global_max_followers = 1.0
        self.global_max_edits = 1.0
        self.valid_tracks = set()

    def precompute_stats(self, data_path: str, slice_files: List[str]):
        """
        Pass 1: Go through all files to get global max stats and track counts.
        """
        print("Pass 1: Pre-computing global stats and track counts...")
        
        track_counts = Counter()
        max_followers = 0
        max_edits = 0

        for slice_file in tqdm(slice_files, desc="Pre-computing"):
            file_path = os.path.join(data_path, slice_file)
            try:
                with open(file_path, 'r') as f:
                    slice_data = json.load(f)
                
                for playlist in slice_data['playlists']:
                    if playlist['num_tracks'] < self.min_playlist_tracks:
                        continue
                    
                    # Update global max stats
                    followers = playlist.get('num_followers', 0)
                    edits = playlist.get('num_edits', 1)
                    if followers > max_followers:
                        max_followers = followers
                    if edits > max_edits:
                        max_edits = edits
                    
                    # Update track counts
                    for track in playlist['tracks']:
                        track_counts[track['track_uri']] += 1
                        
            except json.JSONDecodeError as e:
                print(f"\n[Warning] Failed to parse {slice_file}: {e}. Skipping.")
        
        # Store global stats
        self.global_max_followers = max_followers if max_followers > 0 else 1.0
        self.global_max_edits = max_edits if max_edits > 0 else 1.0

        # Create the set of valid tracks
        self.valid_tracks = {
            track_uri for track_uri, count in track_counts.items() 
            if count >= self.min_track_occurrences
        }
        
        print(f"\nGlobal Max Followers: {self.global_max_followers}")
        print(f"Global Max Edits: {self.global_max_edits}")
        print(f"Total unique tracks found: {len(track_counts):,}")
        print(f"Valid tracks (>= {self.min_track_occurrences} occurrences): {len(self.valid_tracks):,}")

    def process_and_save_batches(self, data_path: str, slice_files: List[str], 
                                 batch_size: int, temp_output_dir: str):
        """
        Pass 2: Load, filter, process, and save data in batches.
        """
        print(f"Pass 2: Processing data in {batch_size}-file batches...")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        num_batches = int(np.ceil(len(slice_files) / batch_size))
        
        for i in tqdm(range(num_batches), desc="Processing Batches"):
            batch_slice_files = slice_files[i*batch_size : (i+1)*batch_size]
            all_interactions = []
            
            for slice_file in batch_slice_files:
                file_path = os.path.join(data_path, slice_file)
                try:
                    with open(file_path, 'r') as f:
                        slice_data = json.load(f)
                    
                    for playlist in slice_data['playlists']:
                        if playlist['num_tracks'] < self.min_playlist_tracks:
                            continue
                        
                        playlist_id = playlist['pid']
                        num_followers = playlist.get('num_followers', 0)
                        num_edits = playlist.get('num_edits', 1)
                        modified_at = playlist.get('modified_at', 0) 
                        
                        for track in playlist['tracks']:
                            # Filter
                            if track['track_uri'] not in self.valid_tracks:
                                continue
                            
                            interaction = {
                                'playlist_id': playlist_id,
                                'track_uri': track['track_uri'],
                                'track_name': track['track_name'],
                                'artist_uri': track['artist_uri'],
                                'artist_name': track['artist_name'],
                                'album_uri': track['album_uri'],
                                'album_name': track['album_name'],
                                'position': track['pos'],
                                'num_followers': num_followers,
                                'num_edits': num_edits,
                                'modified_at': modified_at, 
                                'duration_ms': track.get('duration_ms', 0) 
                            }
                            all_interactions.append(interaction)
                
                except json.JSONDecodeError:
                    continue
            
            if not all_interactions:
                continue # Skip empty batches
                
            # Convert to DataFrame
            batch_df = pd.DataFrame(all_interactions)
            
            # Apply weighting (using global stats)
            batch_df = self.apply_interaction_weighting(batch_df)
            
            # Save processed batch
            batch_df.to_parquet(
                os.path.join(temp_output_dir, f'batch_{i}.parquet'), 
                index=False
            )

    def apply_interaction_weighting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weighting. uses pre-computed global max values.
        """
        # Position weight
        position_weight = np.exp(-self.position_weight_decay * df['position'])
        
        # Follower weight 
        follower_weight = np.log1p(df['num_followers']) / np.log1p(self.global_max_followers)
        follower_weight = 0.5 + follower_weight
        
        # Edit weight
        edit_weight = np.log1p(df['num_edits']) / np.log1p(self.global_max_edits)
        edit_weight = 0.5 + edit_weight
        
        df['interaction_weight'] = position_weight * follower_weight * edit_weight
        
        return df

    def finalize_processing(self, temp_dir: str, output_dir: str):
        """
        Pass 3: Combine batches, fit mappers, split, and save.
        """
        print("Pass 3: Finalizing processing...")
        batch_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.parquet')]
        
        if not batch_files:
            print("Error: No batch files found. Aborting.")
            return

        print("Fitting encoders...")

        # Load only the ID columns from all batches to fit encoders
        all_playlist_ids = pd.concat(
            [pd.read_parquet(f, columns=['playlist_id']) for f in batch_files]
        )['playlist_id'].unique()
        
        all_track_uris = pd.concat(
            [pd.read_parquet(f, columns=['track_uri']) for f in batch_files]
        )['track_uri'].unique()
        
        self.playlist_encoder.fit(all_playlist_ids)
        self.track_encoder.fit(all_track_uris)
        
        playlist_mapping = dict(zip(self.playlist_encoder.classes_, range(len(self.playlist_encoder.classes_))))
        track_mapping = dict(zip(self.track_encoder.classes_, range(len(self.track_encoder.classes_))))
        print(f"Created mappings for {len(playlist_mapping):,} playlists and {len(track_mapping):,} tracks")

        # Load all processed data 
        print("Loading all processed batches...")

        try:
            df = pd.read_parquet(batch_files)
        except MemoryError:
            print("\n--- CRITICAL: Out of Memory ---")
            print("Even the filtered dataset is too large to load into RAM.")
            print("Consider using a library like Dask or Polars for out-of-core processing.")
            return

        # Apply Mappings and Final Weight Norm 
        print("Applying mappings and final normalization...")
        df['playlist_idx'] = self.playlist_encoder.transform(df['playlist_id'])
        df['track_idx'] = self.track_encoder.transform(df['track_uri'])
        
        # Normalize weights by global mean
        mean_weight = df['interaction_weight'].mean()
        if mean_weight > 0:
            df['interaction_weight'] = df['interaction_weight'] / mean_weight
        
        print(f"Weight statistics (Global):")
        print(f"  Min: {df['interaction_weight'].min():.4f}")
        print(f"  Max: {df['interaction_weight'].max():.4f}")
        print(f"  Mean: {df['interaction_weight'].mean():.4f}")
        print(f"  Std: {df['interaction_weight'].std():.4f}")


        track_metadata = self.extract_track_metadata(df)

        # Split into train, val, test sets for later on
        train_df, val_df, test_df = self.temporal_train_test_split(df)
        

        self.save_processed_data(
            train_df, val_df, test_df, track_metadata,
            playlist_mapping, track_mapping, output_dir
        )
        

        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)



    def temporal_train_test_split(self, df: pd.DataFrame, 
                                  train_ratio: float = 0.8, 
                                  val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Performing temporal train/val/test split...")
        df_sorted = df.sort_values(by=['modified_at', 'playlist_id']).copy()
        
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
        test_df = df_sorted.iloc[n_train + n_val:].copy()
        
        print(f"Split sizes:")
        print(f"  Train: {len(train_df):,} interactions ({len(train_df)/len(df_sorted)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} interactions ({len(val_df)/len(df_sorted)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} interactions ({len(test_df)/len(df_sorted)*100:.1f}%)")
        return train_df, val_df, test_df
    
    def extract_track_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Extracting track metadata...")
        track_metadata = df[[
            'track_idx', 'track_uri', 'track_name', 
            'artist_uri', 'artist_name', 
            'album_uri', 'album_name', 
            'duration_ms'
        ]].drop_duplicates(subset=['track_idx']).sort_values('track_idx').reset_index(drop=True)
        print(f"Extracted metadata for {len(track_metadata):,} unique tracks")
        return track_metadata
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, track_metadata: pd.DataFrame,
                           playlist_mapping: Dict, track_mapping: Dict,
                           output_dir: str = 'data/processed'):
        print(f"Saving processed data to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
        val_df.to_parquet(os.path.join(output_dir, 'val.parquet'), index=False)
        test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
        track_metadata.to_parquet(os.path.join(output_dir, 'track_metadata.parquet'), index=False)
        
        with open(os.path.join(output_dir, 'playlist_mapping.pkl'), 'wb') as f:
            pickle.dump(playlist_mapping, f)
        with open(os.path.join(output_dir, 'track_mapping.pkl'), 'wb') as f:
            pickle.dump(track_mapping, f)
        with open(os.path.join(output_dir, 'playlist_encoder.pkl'), 'wb') as f:
            pickle.dump(self.playlist_encoder, f)
        with open(os.path.join(output_dir, 'track_encoder.pkl'), 'wb') as f:
            pickle.dump(self.track_encoder, f)
        
        print("Processed data saved successfully!")
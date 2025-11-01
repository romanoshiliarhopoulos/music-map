import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Iterator, Tuple, List
import torch
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq


class Preprocessor:
    def __init__(self, chunk_size: int = 100000, min_interactions: int = 20):
        self.user_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()
        self.chunk_size = chunk_size
        self.min_interactions = min_interactions

    def _read_parquet_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Read parquet file in chunks using PyArrow"""
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            chunk_df = batch.to_pandas()
            yield chunk_df

    def load_and_merge_data_chunked(self, data_path: str) -> pd.DataFrame:
            """Load and combine all interaction files using chunked processing"""
            print("Loading Yambda-5B interactions in chunks...")
            
            all_interactions = []
            
            # Process listens file in chunks
            print("Processing listens file...")
            for chunk_df in self._read_parquet_chunks(f"{data_path}/listens.parquet"):
                chunk_df.rename(columns={'uid': 'user_id', 'item_id': 'track_id'}, inplace=True)
                chunk_df['interaction_type'] = 'listen'
                chunk_df['weight'] = (chunk_df['played_ratio_pct'] / 100.0).astype('float32')
                chunk_df['play_count'] = 1
                
                #-optimize data type to reduce memory
                chunk_df['interaction_type'] = chunk_df['interaction_type'].astype('category')
                chunk_df['play_count'] = chunk_df['play_count'].astype('int16')
                
                selected_cols = ['user_id', 'track_id', 'timestamp', 'is_organic', 'interaction_type', 'weight', 'play_count']
                all_interactions.append(chunk_df[selected_cols])
                
            # Process likes file in chunks  
            print("Processing likes file...")
            for chunk_df in self._read_parquet_chunks(f"{data_path}/likes.parquet"):
                chunk_df.rename(columns={'uid': 'user_id', 'item_id': 'track_id'}, inplace=True)
                chunk_df['interaction_type'] = 'like'
                chunk_df['weight'] = 5.0
                chunk_df['play_count'] = 1

                # -optimize data type to reduce memory
                chunk_df['interaction_type'] = chunk_df['interaction_type'].astype('category')
                chunk_df['weight'] = chunk_df['weight'].astype('float32')
                chunk_df['play_count'] = chunk_df['play_count'].astype('int16')

                selected_cols = ['user_id', 'track_id', 'timestamp', 'is_organic', 'interaction_type', 'weight', 'play_count']
                all_interactions.append(chunk_df[selected_cols])
            
            # Process dislikes file in chunks 
            print("Processing dislikes file...")
            for chunk_df in self._read_parquet_chunks(f"{data_path}/dislikes.parquet"):
                chunk_df.rename(columns={'uid': 'user_id', 'item_id': 'track_id'}, inplace=True)
                chunk_df['interaction_type'] = 'dislike' 
                chunk_df['weight'] = -3.0
                chunk_df['play_count'] = 1

                # -optimize data type to reduce memory
                chunk_df['interaction_type'] = chunk_df['interaction_type'].astype('category')
                chunk_df['weight'] = chunk_df['weight'].astype('float32')
                chunk_df['play_count'] = chunk_df['play_count'].astype('int16')
                
                selected_cols = ['user_id', 'track_id', 'timestamp', 'is_organic', 'interaction_type', 'weight', 'play_count']
                all_interactions.append(chunk_df[selected_cols])
            
            # Combine all interactions
            print("Combining interactions...")
            combined_interactions = pd.concat(all_interactions, ignore_index=True)
            
            # Free memory
            del all_interactions

            print(f"Combined dataset: {len(combined_interactions)} interactions")
            
            return combined_interactions

    def create_user_item_mappings(self, df: pd.DataFrame) -> Tuple[Dict, Dict, pd.DataFrame]:
        """Create user and item ID mappings with filtering"""
        print("Creating user-item mappings...")
        
        # Count interactions per user and item
        user_counts = df['user_id'].value_counts()
        item_counts = df['track_id'].value_counts()
        
        # Filter users and items with minimum interactions
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_interactions].index
        
        # Filter dataframe
        print(f"Filtering {len(df)} interactions by user...")
        df_filtered = df[df['user_id'].isin(valid_users)]

        print(f"Filtering {len(df_filtered)} interactions by item...")
        df_filtered = df_filtered[df_filtered['track_id'].isin(valid_items)].copy()
        
        print(f"Filtered from {len(df)} to {len(df_filtered)} interactions")
        print(f"Users: {len(df['user_id'].unique())} -> {len(valid_users)}")
        print(f"Items: {len(df['track_id'].unique())} -> {len(valid_items)}")
        
        # Create encoded indices
        df_filtered['user_idx'] = self.user_encoder.fit_transform(df_filtered['user_id'])
        df_filtered['item_idx'] = self.song_encoder.fit_transform(df_filtered['track_id'])
        
        # Create mappings
        user_mapping = dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_))))
        item_mapping = dict(zip(self.song_encoder.classes_, range(len(self.song_encoder.classes_))))
        
        return user_mapping, item_mapping, df_filtered

    def temporal_train_test_split(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally to prevent data leakage"""
        
        print("Creating temporal train/validation/test splits...")
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Calculate split sizes
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create splits
        train_df = df_sorted[:n_train].copy()
        val_df = df_sorted[n_train:n_train + n_val].copy()
        test_df = df_sorted[n_train + n_val:].copy()
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

    def process_and_save(self, data_path: str, output_dir: str = "data/processed") -> str:
        """Process data and save all intermediate results"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        processed_file = f"{output_dir}/processed_interactions.parquet"
        mappings_file = f"{output_dir}/user_item_mappings.pkl"
        splits_dir = f"{output_dir}/splits"
        
        # Check if already processed
        if os.path.exists(processed_file) and os.path.exists(mappings_file):
            print(f"Loading existing preprocessed data from {output_dir}")
            return self.load_preprocessed_data(output_dir)
        
        print("Processing raw data...")
        
        # Load and merge data
        df = self.load_and_merge_data_chunked(data_path)
        
        # Create mappings and filter
        user_mapping, item_mapping, df_filtered = self.create_user_item_mappings(df)
        
        # Save processed interactions
        print(f"Saving processed data to {processed_file}")
        df_filtered.to_parquet(processed_file, index=False, compression='snappy')
        
        # Save mappings (pickle for Python objects)
        import pickle
        mappings = {
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'user_encoder': self.user_encoder,
            'song_encoder': self.song_encoder
        }
        
        with open(mappings_file, 'wb') as f:
            pickle.dump(mappings, f)
        
        # Create and save train/val/test splits
        train_df, val_df, test_df = self.temporal_train_test_split(df_filtered)
        
        Path(splits_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(f"{splits_dir}/train.parquet", index=False)
        val_df.to_parquet(f"{splits_dir}/val.parquet", index=False) 
        test_df.to_parquet(f"{splits_dir}/test.parquet", index=False)
        
        # Save metadata
        metadata = {
            'total_interactions': len(df_filtered),
            'num_users': len(user_mapping),
            'num_items': len(item_mapping), 
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'min_interactions': self.min_interactions,
            'chunk_size': self.chunk_size
        }
        
        import json
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Preprocessing complete! Data saved to {output_dir}")
        print(f"Dataset stats: {metadata}")
        
        return output_dir
    
    def load_preprocessed_data(self, output_dir: str):
        """Load existing preprocessed data"""
        
        # Load processed interactions
        df = pd.read_parquet(f"{output_dir}/processed_interactions.parquet")
        
        # Load mappings
        import pickle
        with open(f"{output_dir}/user_item_mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)
            
        # Load splits
        splits = {
            'train': pd.read_parquet(f"{output_dir}/splits/train.parquet"),
            'val': pd.read_parquet(f"{output_dir}/splits/val.parquet"),
            'test': pd.read_parquet(f"{output_dir}/splits/test.parquet")
        }
        
        # Load metadata
        import json
        with open(f"{output_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded preprocessed data: {metadata}")
        
        return {
            'data': df,
            'mappings': mappings, 
            'splits': splits,
            'metadata': metadata
        }

    def get_data_stats(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_interactions': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_items': df['track_id'].nunique(),
            'sparsity': 1.0 - (len(df) / (df['user_id'].nunique() * df['track_id'].nunique())),
            'avg_interactions_per_user': len(df) / df['user_id'].nunique(),
            'avg_interactions_per_item': len(df) / df['track_id'].nunique(),
            'interaction_types': df['interaction_type'].value_counts().to_dict(),
            'organic_ratio': df['is_organic'].mean() if 'is_organic' in df.columns else None,
            'weight_distribution': {
                'min': df['weight'].min(),
                'max': df['weight'].max(),
                'mean': df['weight'].mean(),
                'std': df['weight'].std()
            }
        }
        return stats

# Cold-Run pre-processor
def main():
    print("STARTING PREPROCESSING!")
    # Initialize preprocessor
    preprocessor = Preprocessor(chunk_size=100000, min_interactions=10)
    
    raw_data_path = "data/raw"  
    output_path = "data/processed"  
    
    # Run preprocessing
    try:
        result_path = preprocessor.process_and_save(raw_data_path, output_path)
        print(f"✅ Preprocessing completed successfully!")
        print(f"📁 Processed data saved to: {result_path}")
        
        # Load and verify the results
        data = preprocessor.load_preprocessed_data(output_path)
        print(f"📊 Dataset stats: {data['metadata']}")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()

import os
import pandas as pd

def inspect_data_structure(data_path: str):
    """Inspect the structure of parquet files to understand available columns"""
    files_to_inspect = ['listens.parquet', 'likes.parquet', 'dislikes.parquet', 'embeddings.parquet']
    
    for filename in files_to_inspect:
        file_path = f"{data_path}/{filename}"
        if os.path.exists(file_path):
            print(f"\n📋 {filename} structure:")
            # Read just the first few rows to see columns
            sample = pd.read_parquet(file_path)
            print(f"   Columns: {list(sample.columns)}")
            print(f"   Shape: {sample.shape}")
            print(f"   Sample data:")
            print(sample.head(2).to_string())
        else:
            print(f"{filename} not found")

inspect_data_structure("data/raw")
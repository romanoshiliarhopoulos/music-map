import os
from datasets import load_dataset
import pandas as pd
import gc # Make sure garbage collector is imported
from tqdm import tqdm

def download_yambda_50M():
    """Downloads yambda 50M from HuggingFace """

    print("Downloading Yambda-50M dataset ...")

    data_dir = "flat/50m"  
    
    dataset_files = [
        "listens.parquet",
        "likes.parquet", 
        "dislikes.parquet"
    ]

    interaction_track_ids = set()
    
    for file in dataset_files:
        print(f"Downloading file: {file} ...")
        try:
            data = load_dataset("yandex/yambda", 
                              data_dir=data_dir, 
                              data_files=file, 
                              split="train")
            
            df = data.to_pandas()
            
            if 'item_id' in df.columns:
                track_ids = set(df['item_id'].unique())
                interaction_track_ids.update(track_ids)
                print(f"Found {len(track_ids):,} unique tracks in {file}")
            
            os.makedirs("data/raw", exist_ok=True)
            df.to_parquet(f"data/raw/{file}", index=False)
            print(f"Saved {len(df)} records to data/raw/{file}")
            
            del df, data
            gc.collect() # Force garbage collection
            
        except Exception as e:
            print(f"Error downloading {file}: {e}")
    
    print(f"\nTotal unique tracks in 50M dataset: {len(interaction_track_ids):,}")
    
    print("\nDownloading and filtering embeddings.parquet ...")
    download_filtered_embeddings(interaction_track_ids)

import dask.dataframe as dd 

def download_filtered_embeddings(required_track_ids):
    """
    Download embeddings using streaming, filters each item before batching,
    and saves filtered batches to disk to avoid OOM.
    """
    
    output_filename = "data/raw/embeddings.parquet"
    temp_dir = "data/temp_batches"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("Loading embeddings with streaming...")
        
        embeddings_dataset = load_dataset("yandex/yambda", 
                                         data_files="embeddings.parquet", 
                                         split="train",
                                         streaming=True)
        

        save_batch_size = 10000 
        
        total_processed = 0
        found_embeddings = 0
        filtered_items_batch = []
        batch_file_paths = []

        print("Filtering embeddings and saving batches with aggressive memory cleanup...")
        
        for item in tqdm(embeddings_dataset, desc="Processing embeddings", unit=" rows"):
            total_processed += 1
            
            if item['item_id'] in required_track_ids:
                # Convert to a basic dict 
                filtered_items_batch.append({'item_id': item['item_id'], 'normalized_embed': item['normalized_embed']})
                found_embeddings += 1
            
            if len(filtered_items_batch) >= save_batch_size:
                
                # We pass the count to save_batch_to_parquet now
                save_batch_to_parquet(filtered_items_batch, temp_dir, len(batch_file_paths))
                batch_file_paths.append(1) 
                
                del filtered_items_batch
                filtered_items_batch = []
                gc.collect()
        
        # Save the final batch
        if filtered_items_batch:
            save_batch_to_parquet(filtered_items_batch, temp_dir, len(batch_file_paths))
            batch_file_paths.append(1)
            del filtered_items_batch
            gc.collect()


        if batch_file_paths:
            print(f"\nCombining {len(batch_file_paths)} temporary batches into final embeddings.parquet using Dask...")
            
            dask_df = dd.read_parquet(f"{temp_dir}/batch_*.parquet")
            
            dask_df.to_parquet(output_filename, write_index=False, schema="infer")

            print(f"Successfully combined batches into {output_filename}")
            
            print("Cleaning up temporary batch files...")

            import glob
            for f in glob.glob(f"{temp_dir}/batch_*.parquet"):
                os.remove(f)
            os.rmdir(temp_dir)
            
        else:
            print("No matching embeddings found!")
            
    except Exception as e:
        print(f"Error downloading embeddings: {e}")

def save_batch_to_parquet(batch_data, temp_dir, batch_index):
    """
    Converts a list of filtered items to a DataFrame, saves it,
    and cleans up the DataFrame from memory.
    """
    try:
        df = pd.DataFrame(batch_data)
        
        
        if df['item_id'].dtype == 'int64':
            df['item_id'] = pd.to_numeric(df['item_id'], downcast='signed')
        
        batch_path = os.path.join(temp_dir, f"batch_{batch_index:05d}.parquet")
        df.to_parquet(batch_path, index=False)
                
        del df
        
    except Exception as e:
        print(f"Error saving batch: {e}")
        
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    download_yambda_50M()
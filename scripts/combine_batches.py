import pandas as pd
import glob
import os

# Find all parquet part files
parquet_files = glob.glob("data/raw/embeddings.parquet/part.*.parquet")

# Sort numerically by part number
parquet_files.sort(key=lambda x: int(os.path.basename(x).replace("part.", "").replace(".parquet", "")))

print(f"Combining {len(parquet_files)} parquet files...")

# Read and concatenate all files
dfs = [pd.read_parquet(f) for f in parquet_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Save combined file
output_path = "data/raw/embeddings.parquet"
combined_df.to_parquet(output_path, index=False)

print(f"Created {output_path} with shape {combined_df.shape}")

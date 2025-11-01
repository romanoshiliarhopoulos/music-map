import pandas as pd
df = pd.read_parquet('data/raw/embeddings.parquet')

print(f"Shape: {df.shape}")           
print(f"Columns: {df.columns.tolist()}")  
print(f"Data types:\n{df.dtypes}")    
print(df.head()) 
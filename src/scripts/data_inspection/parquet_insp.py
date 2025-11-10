import pandas as pd
import pyarrow.parquet as pq
import sys
import os
from typing import List

FILE_PATH = "data/processed/test.parquet"

VALUE_TO_FIND = 319783
# ---------------------

def search_parquet(filepath: str, value: any):
    """
    Searches a Parquet file chunk by chunk for an exact value.
    This is memory-efficient and suitable for large files.
    """
    
    search_values = [value, str(value)]
    
    print(f"--- Searching for {search_values} in {filepath} ---")

    try:
        # Open the Parquet file metadata (doesn't load data)
        pf = pq.ParquetFile(filepath)
    except FileNotFoundError:
        print(f"\nError: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred trying to open the file: {e}")
        sys.exit(1)

    matching_rows_list = []
    total_matches = 0

    print(f"File has {pf.num_row_groups} row group(s). Scanning now...")

    for i in range(pf.num_row_groups):
        print(f"  Scanning chunk {i + 1}/{pf.num_row_groups}...")
        
        table = pf.read_row_group(i)
        df = table.to_pandas()

        for col in df.columns:
            try:

                matches_mask = df[col].isin(search_values)
                
                if matches_mask.any():
                    num_found = matches_mask.sum()
                    total_matches += num_found
                    print(f"    -> Found {num_found} match(es) in column: '{col}'")
                    
                    matching_rows_list.append(df[matches_mask])

            except (TypeError, AttributeError):
                pass
            except Exception as e:
                print(f"    -> Error searching column {col}: {e}")

    if not matching_rows_list:
        print("\n---No exact matches found. ---")
        return

    print(f"\n---Found {total_matches} total match(es) ---")
    
    # Combine all the small DataFrames of matching rows into one
    all_matching_rows = pd.concat(matching_rows_list)
    
    # Drop duplicates in case the same row was found for different reasons
    all_matching_rows = all_matching_rows.drop_duplicates()

    print(f"Displaying {len(all_matching_rows)} unique matching rows:")
    pd.set_option('display.max_columns', None)
    print(all_matching_rows)

def print_parquet(filepath: str):
    """
    Reads a Parquet file and prints its structure (shape, schema, and head).
    """
    
    print(f"---Inspecting: {filepath} ---")

    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return
        
    try:
        df = pd.read_parquet(filepath)

        print(f"\nShape: {df.shape} (Rows, Columns)")

        print("\nSchema (df.info()):")
        df.info()

        print("\nHead (First 5 Rows):")
        pd.set_option('display.max_columns', None)
        print(df.head())

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")
        print("   The file might be corrupted or not a valid Parquet file.")

    print("\n" + "-" * (len(filepath) + 20))



if __name__ == "__main__":
    print_parquet(FILE_PATH)
    search_parquet(FILE_PATH, VALUE_TO_FIND)
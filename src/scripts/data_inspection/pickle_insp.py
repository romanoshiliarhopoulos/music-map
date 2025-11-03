import pickle
import sys

# --- Configuration ---
FILE_PATH = "data/processed/track_mapping.pkl"
# ---------------------

def load_and_access(filepath: str):
    """
    Loads the pickled dictionary and demonstrates how to access values.
    """
    print(f"--- Loading: {filepath} ---")

    try:
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
            
            if not isinstance(data_dict, dict):
                print(f"Error: Expected a dictionary, but found {type(data_dict)}")
                return
        
        print(f"Successfully loaded dictionary with {len(data_dict)} items.")

        key_to_check = 'spotify:track:000VZqvXwT0YNqKk7iG2GS' 
        if key_to_check in data_dict:
            value = data_dict[key_to_check]
            print(f"\n--- 1. Value for a specific key ---")
            print(f"  Key: {key_to_check}")
            print(f"  Value: {value}")
        else:
            print(f"\nKey '{key_to_check}' not found in dictionary.")

        all_values = list(data_dict.values())
        print(f"\n--- 2. All values ---")
        print(f"  Total values: {len(all_values)}")
        print(f"  First 10 values: {all_values[:10]}")
        
        print(f"\n--- 3. Loop over first 5 key-value pairs ---")
        count = 0
        for key, value in data_dict.items():
            print(f"  '{key}': {value}")
            count += 1
            if count >= 5:
                break

    except FileNotFoundError:
        print(f"\nError: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    load_and_access(FILE_PATH)
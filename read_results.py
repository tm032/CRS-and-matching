import json
import pickle
import os
from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent

# Iterate through all files in the directory
for file_path in current_dir.iterdir():
    if file_path.suffix == '.json':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                alpha = data.get('alpha', 'N/A')
                print(f"{file_path.name}: {alpha}")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    elif file_path.suffix == '.pkl':
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                alpha = data.get('alpha', 'N/A')
                print(f"{file_path.name}: {alpha}")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
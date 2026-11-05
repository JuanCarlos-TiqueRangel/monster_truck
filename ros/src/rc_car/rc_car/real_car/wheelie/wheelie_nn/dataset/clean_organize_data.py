import os
import pandas as pd
from pathlib import Path


# Define the folder directory containing your files
# Change this path to your actual folder location (e.g., "C:/data/files" or "/home/user/data")

BASE_DIR = Path(__file__).resolve().parent
folder_path = BASE_DIR #/ "dataset"

# Total sequence bounds from rls_id_001 to rls_id_147
start_id = 1
end_id = 147

print("Starting file filtering process...\n")

# Loop through the sequence from 1 to 147
for i in range(start_id, end_id + 1):
    # Formats the index with leading zeros (e.g., 1 -> '001', 12 -> '012')
    file_name = f"rls_id_{i:03d}"
    
    # Check for the file (assuming it's a .csv file, update extension if it's different)
    file_path = os.path.join(folder_path, f"{file_name}.csv")
    
    if os.path.exists(file_path):
        try:
            # 1. Read the file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Record original row count to measure the change
            original_rows = len(df)
            
            # 2. Filter out rows where sequence_index is -1
            # We keep rows where sequence_index is NOT equal to -1
            filtered_df = df[df['sequence_index'] != -1]
            
            # 3. Save the clean data back to the same file
            filtered_df.to_csv(file_path, index=False)
            
            removed_count = original_rows - len(filtered_df)
            print(f"Processed {file_name}.csv: Removed {removed_count} rows.")
            
        except Exception as e:
            print(f"Error processing {file_name}.csv: {e}")
    else:
        print(f"Warning: File {file_name}.csv not found in folder.")

print("\nProcess completed successfully!")

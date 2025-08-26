import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import gc
import time
import traceback

# Define dataset folder path
DATASET_FOLDER = "../dataset"

# Define column types globally
DTYPE_MAPPINGS = {
    'proto': str,
    'flgs': str,
    'state': str,
    'category': str,
    'attack': str,
    'subcategory': str,
    'pkts': int,
    'bytes': int,
    'seq': int,
    'dur': float,
    'mean': float,
    'stddev': float,
    'sum': float,
    'min': float,
    'max': float,
    'rate': float,
    'srate': float,
    'drate': float
}

def process_file(file_path, chunk_size=10000):
    """Process a single file with better error handling and progress tracking"""
    try:
        print(f"\nReading file: {os.path.basename(file_path)}")
        chunks = []
        processed_rows = 0
        
        # Get file size for progress calculation
        file_size = os.path.getsize(file_path)
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=DTYPE_MAPPINGS, low_memory=False):
            chunks.append(chunk)
            processed_rows += len(chunk)
            progress = (processed_rows / (file_size/chunk_size)) * 100
            progress = min(100, progress)  # Cap at 100%
            print(f"\rProgress: {progress:.2f}% ", end="", flush=True)
            gc.collect()
            
        print("\nConcatenating chunks...")
        result = pd.concat(chunks, ignore_index=True)
        print(f"File processed successfully. Shape: {result.shape}")
        return result
    
    except Exception as e:
        print(f"\nError processing {file_path}: {str(e)}")
        return None

# List all files to merge
all_files = [f"data_{i}.csv" for i in range(1, 3)]  # Reduced to 2 files for testing
temp_files = []

# Process each file individually
for idx, file_name in enumerate(all_files, 1):
    try:
        print(f"\n{'='*50}")
        print(f"Processing file {idx}/{len(all_files)}: {file_name}")
        print(f"{'='*50}")
        
        file_path = os.path.join(DATASET_FOLDER, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        start_time = time.time()
        temp_df = process_file(file_path)
        
        if temp_df is not None:
            temp_filename = os.path.join(DATASET_FOLDER, f'temp_processed_{len(temp_files)}.csv')
            print(f"Saving temporary file: {os.path.basename(temp_filename)}")
            temp_df.to_csv(temp_filename, index=False)
            temp_files.append(temp_filename)
            
            # Clear memory
            del temp_df
            gc.collect()
            
            end_time = time.time()
            print(f"✅ Completed processing {file_name} in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error processing {file_name}: {str(e)}")
        continue

print("\n✅ All files processed. Merging temp files...")

# Modified preprocessing section
try:
    print("\nMerging and preprocessing dataset...")
    final_df = pd.DataFrame()
    
    # Verify temp_files is not empty
    if not temp_files:
        raise ValueError("No temporary files found for processing")
    
    # Check and print column names from first file
    print(f"Reading temporary files: {len(temp_files)} files found")
    first_file = pd.read_csv(temp_files[0], nrows=1)
    available_columns = first_file.columns.tolist()
    print("\nAvailable columns:", available_columns)
    
    # Process each temporary file
    total_processed = 0
    for temp_file in temp_files:
        try:
            print(f"\nProcessing {os.path.basename(temp_file)}...")
            current_chunk = pd.read_csv(temp_file, dtype=DTYPE_MAPPINGS, low_memory=False)
            
            if final_df.empty:
                final_df = current_chunk
            else:
                final_df = pd.concat([final_df, current_chunk], ignore_index=True)
            
            total_processed += len(current_chunk)
            print(f"Processed rows so far: {total_processed:,}")
            
            # Clear memory
            del current_chunk
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {os.path.basename(temp_file)}: {str(e)}")
            continue
    
    if final_df.empty:
        raise ValueError("No data was successfully processed")

    print(f"\nMerged dataset shape: {final_df.shape}")

    
    # Strip whitespace from column names
    final_df.columns = final_df.columns.str.strip()

    feature_cols = [col for col in final_df.columns if col not in ['category', 'subcategory']]
    
    # Drop unnecessary columns if they exist
    drop_cols = ['pkSeqID', 'stime', 'ltime', 'smac', 'dmac', 'saddr', 'sport', 'daddr', 'dport', 'attack']
    existing_cols = [col for col in drop_cols if col in final_df.columns]
    if existing_cols:
        final_df.drop(columns=existing_cols, inplace=True)
        print(f"Dropped columns: {existing_cols}")

    # Handle missing values
    final_df.fillna(0, inplace=True)

    # Encode categorical features
    categorical_cols = ['proto', 'flgs', 'state']
    existing_cats = [col for col in categorical_cols if col in final_df.columns]
    if existing_cats:
        final_df[existing_cats] = final_df[existing_cats].astype(str)
        final_df = pd.get_dummies(final_df, columns=existing_cats)
        print(f"Encoded categorical columns: {existing_cats}")

    # Encode attack labels
    label_encoder = LabelEncoder()
    if 'category' in final_df.columns:
        final_df['category'] = label_encoder.fit_transform(final_df['category'])
    else:
        raise ValueError("'category' column not found in the dataset")
    
    # Separate features and target (excluding category and subcategory if present)
    feature_cols = [col for col in final_df.columns if col not in ['category', 'subcategory']]
    X = final_df[feature_cols]
    y = final_df['category']

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify numeric columns for scaling
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    print(f"\nNumeric features to scale: {len(numeric_features)}")
    
    # Create copies of the datasets
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    
    # Scale only numeric features
    scaler = StandardScaler()
    X_train_df[numeric_features] = scaler.fit_transform(X_train_df[numeric_features])
    X_test_df[numeric_features] = scaler.transform(X_test_df[numeric_features])

    # Save processed data
    print("\nSaving processed datasets...")
    X_train_df.to_csv(os.path.join(DATASET_FOLDER, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(DATASET_FOLDER, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(DATASET_FOLDER, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(DATASET_FOLDER, "y_test.csv"), index=False)

    print("✅ Data preprocessing complete! Processed files saved in dataset folder.")

except Exception as e:
    print(f"Error during final processing: {str(e)}")
    import traceback
    print(traceback.format_exc())  # Print detailed error information

finally:
    # Cleanup temporary files
    print("\nCleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Removed: {os.path.basename(temp_file)}")
        except Exception as e:
            print(f"Failed to remove {os.path.basename(temp_file)}: {str(e)}")

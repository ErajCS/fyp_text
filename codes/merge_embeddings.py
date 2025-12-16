import os
import glob
import numpy as np
import pandas as pd
import re
import chardet

# ---------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------

def detect_encoding(filepath):
    """Detect file encoding using chardet."""
    with open(filepath, 'rb') as f:
        raw_data = f.read(100000)
        result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8-sig'


def read_csv_safely(filepath):
    """Read CSV file safely with detected encoding."""
    enc = detect_encoding(filepath)
    try:
        df = pd.read_csv(filepath, encoding=enc)
    except Exception as e:
        print(f"⚠️ Error reading {filepath} with {enc}: {e}")
        df = pd.read_csv(filepath, encoding='utf-8-sig', on_bad_lines='skip')
    return df


def clean_bidi_chars(text):
    """Remove invisible bidi and formatting characters (for Urdu text)."""
    if isinstance(text, str):
        # Removes RLE, LRE, PDF, etc., and zero-width characters
        return re.sub(r'[\u202A-\u202E\u200B-\u200F]', '', text)
    return text


def merge_embeddings(language):
    """Merge CSV and NPY batches for a given language."""
    print(f"\n🔹 Merging {language.capitalize()} embeddings...")

    csv_pattern = f"embeddings_output/{language}_embeddings_batch_*.csv"
    npy_pattern = f"embeddings_output/{language}_vectors_batch_*.npy"

    csv_files = sorted(glob.glob(csv_pattern))
    npy_files = sorted(glob.glob(npy_pattern))

    if not csv_files:
        print(f"⚠️ No CSV files found for {language}. Skipping.")
        return

    # Read all CSVs safely
    dfs = []
    for file in csv_files:
        df = read_csv_safely(file)
        dfs.append(df)

    # Ensure consistent columns across all batches
    base_cols = dfs[0].columns
    for i, df in enumerate(dfs):
        if not all(df.columns == base_cols):
            print(f"⚠️ Column mismatch in {csv_files[i]}")
            print(f"Found columns: {df.columns}")
            dfs[i] = df[base_cols.intersection(df.columns)]

    merged_df = pd.concat(dfs, ignore_index=True)

    # Clean Urdu RTL characters only for Urdu
    if language.lower() == "urdu":
        merged_df = merged_df.applymap(clean_bidi_chars)

    # Merge numpy vector files
    merged_vecs = np.vstack([np.load(f) for f in npy_files])

    # Output folder
    output_dir = "embeddings_output/merged"
    os.makedirs(output_dir, exist_ok=True)

    # Save merged files
    merged_df.to_csv(f"{output_dir}/{language}_embeddings_merged.csv",
                     index=False, encoding="utf-8-sig")
    np.save(f"{output_dir}/{language}_vectors_merged.npy", merged_vecs)

    print(f"✅ {language.capitalize()} merged: {len(merged_df)} records successfully saved.")


# ---------------------------------------------------------
#  Main execution
# ---------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("embeddings_output/merged", exist_ok=True)

    merge_embeddings("english")
    merge_embeddings("urdu")

    print("\n🎉 All embeddings successfully merged and cleaned!")

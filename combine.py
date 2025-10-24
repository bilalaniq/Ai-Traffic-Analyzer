import os
import pandas as pd

# Folder containing all your CSVs
folder = "MachineLearningCVE"

# List all .csv files in folder
csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
print(f"Found {len(csv_files)} CSV files:\n", csv_files)

# Read and combine all
dataframes = []
for file in csv_files:
    path = os.path.join(folder, file)
    print(f"📂 Loading: {file}")
    try:
        df = pd.read_csv(path, low_memory=False)

        # 🧹 Clean column names (remove spaces, special chars)
        df.columns = df.columns.str.strip()

        dataframes.append(df)
    except Exception as e:
        print(f"❌ Error loading {file}: {e}")

# Combine into single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# 🧹 Clean final combined columns too (just to be safe)
combined_df.columns = combined_df.columns.str.strip()

# Drop duplicates (optional but recommended)
combined_df = combined_df.drop_duplicates()

# Save combined dataset
output_file = "combined_cicids2017.csv"
combined_df.to_csv(output_file, index=False)

print(f"\n✅ Combined dataset saved as '{output_file}'")
print(f"🧮 Total Rows: {len(combined_df)} | Columns: {len(combined_df.columns)}")

# Quick preview
print("\n🔹 First 5 rows:")
print(combined_df.head())

print("\n📊 Label distribution:")
label_col = None
for col in combined_df.columns:
    if col.strip().lower() == "label":
        label_col = col
        break

if label_col:
    print(combined_df[label_col].value_counts())
else:
    print("⚠️ No 'Label' column found — check column names:")
    print(combined_df.columns.tolist())

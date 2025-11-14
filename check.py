import pandas as pd
import matplotlib.pyplot as plt

# ✅ Use forward slashes for Linux/WSL
df = pd.read_csv("./combined_cicids2017.csv")

# ✅ Strip spaces from column names (fixes ' Label' issue)
df.columns = df.columns.str.strip()

print(f"Loaded full dataset with {len(df)} rows and {len(df.columns)} columns.\n")

print("🔹 First 5 rows of dataset:")
print(df.head())

print("\n📊 Total rows in dataset:", len(df))

print("\n📈 Class counts:")
print(df["Label"].value_counts())

print("\n📊 Class distribution (percentages):")
print(df["Label"].value_counts(normalize=True) * 100) # normalize=True = gives the fraction (between 0 and 1) of the total.

plt.figure(figsize=(8,4))
df["Label"].value_counts().plot(kind='bar', color=['green', 'red', 'blue', 'orange', 'purple', 'brown'])
plt.yscale('log')  # logarithmic scale
plt.title("Attack vs Normal Traffic (Full Dataset)")
plt.xlabel("Traffic Type")
plt.ylabel("Number of Samples (log scale)")
plt.xticks(rotation=25)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===============================
# Load dataset (with encoding fallback)
# ===============================
print("📂 Loading dataset...")
try:
    df = pd.read_csv("combined_cicids2017.csv", encoding='utf-8')
    print("✅ Loaded dataset using UTF-8 encoding.\n")
except UnicodeDecodeError:
    df = pd.read_csv("combined_cicids2017.csv", encoding='latin1')
    print("⚠️ UTF-8 failed. Loaded using Latin-1 encoding.\n")

print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns\n")

# ===============================
# Clean dataset
# ===============================
# Replace infinities with NaN and drop rows with missing values
df = df.replace([np.inf, -np.inf], np.nan).dropna()
# Remove spaces from column names
df.columns = df.columns.str.strip()

# ===============================
# Show classes
# ===============================
print("\n📊 Classes in dataset:")
print(df["Label"].value_counts())

# ===============================
# Prepare features and labels
# ===============================
X = df.drop(columns=["Label"])             # Features
y = df["Label"].astype(str).str.strip()    # Labels

# Encode text labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Converts text labels to numbers

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n🧠 Training samples:", len(X_train))
print("🧪 Testing samples:", len(X_test))
print("🔢 Total Classes:", len(label_encoder.classes_))
print("Classes:", list(label_encoder.classes_))

# ===============================
# Train model
# ===============================
print("\n⚙️ Training Multi-Class RandomForest...")
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ===============================
# Visualization: All Classes
# ===============================
plt.figure(figsize=(12, 6))
ax = df["Label"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Class Distribution (Multi-Class)", fontsize=14)
plt.xlabel("Traffic Type")
plt.ylabel("Count (log scale)")
plt.yscale("log")
plt.xticks(rotation=45, ha='right')
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Annotate bar heights
for p in ax.patches:     # ax.patches gives a list of all the bars in the chart.
    ax.annotate(f'{int(p.get_height()):,}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=8, rotation=90)
    
    # f'{int(p.get_height()):,}' → formats the number with commas, e.g., 10000 → 10,000
    # (p.get_x() + p.get_width()/2., p.get_height()) → x and y coordinates for the text:
    # ha='center' → horizontal alignment at center
    # va='bottom' → vertical alignment at the bottom of the text (so it sits just above the bar)

plt.tight_layout()  # Automatically adjusts spacing in the figure so labels, titles, and bars don’t overlap.
plt.show()

# ===============================
# Save model and preprocessors
# ===============================
joblib.dump(model, "traffic_detector_multiclass.pkl")
joblib.dump(label_encoder, "label_encoder_multi.pkl")
joblib.dump(scaler, "scaler_multi.pkl")

print("\n💾 Model saved successfully (traffic_detector_multiclass.pkl)")
print("💾 Label encoder saved (label_encoder_multi.pkl)")
print("💾 Scaler saved (scaler_multi.pkl)")

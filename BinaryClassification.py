import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load dataset
# ===============================
print("📂 Loading dataset...")
df = pd.read_csv("combined_cicids2017.csv")   # Reads the combined CICIDS2017 dataset you previously merged.
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns\n")

# ===============================
# Clean dataset
# ===============================
# Drop infinite and missing values
df = df.replace([float("inf"), float("-inf")], None).dropna()

# Check column names
print("📋 Columns in dataset:")
print(df.columns.tolist())

# ===============================
# Create binary labels
# ===============================
df["Label"] = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

print("\n📊 Binary label counts:")
print(df["Label"].value_counts())

# ===============================
# Prepare features & labels
# ===============================
X = df.drop(columns=["Label"])
y = df["Label"]

# Encode target (BENIGN=0, ATTACK=1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # this is where conversion happend

# ===============================
# Split data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("\n🧠 Training samples:", len(X_train))
print("🧪 Testing samples:", len(X_test))

# X_train: Training features
# X_test: Testing features
# y_train: Training labels
# y_test: Testing labels

# ===============================
# Train model
# ===============================
print("\n⚙️ Training RandomForestClassifier...")  # RandomForestClassifier is a machine learning algorithm based on an ensemble of decision trees
# It doesn’t rely on a single tree — instead, it builds many decision trees (a forest) and combines their results to make the final prediction.
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,  # This ensures your results are reproducible every time you run the script.
    n_jobs=-1,        # This tells the model to use all CPU cores available for training.
    class_weight="balanced_subsample"  # If your dataset has way more BENIGN traffic than ATTACK, this helps prevent the model from being biased toward the majority class
)
model.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===============================
# Visualize class distribution
# ===============================
plt.figure(figsize=(6, 4))
df["Label"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Class Distribution (Binary Mode)")
plt.xlabel("Traffic Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# ===============================
# Save model
# ===============================
joblib.dump(model, "traffic_detector_binary.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model saved as traffic_detector_binary.pkl")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
csv_file = "updated_dataset_main.csv"
df = pd.read_csv(csv_file)

# Features and target
features = ['H', 'e', 'albedo', 'ad']
target = 'main_class'

# Encode target
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Prepare data
X = df[features]
y = df[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest model accuracy on test set: {accuracy:.4f}")

# Save model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print("Random Forest model trained and saved.")

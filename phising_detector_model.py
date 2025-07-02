import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv(r'data/Phishing URLs.csv')  # Update with your dataset path

# Function to extract features from the URL
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'has_https': 1 if parsed_url.scheme == 'https' else 0,
        'num_subdomains': url.count('.') - 1,  # Count of subdomains
    }
    return features

# Apply feature extraction
features = df['url'].apply(extract_features).apply(pd.Series)

# Combine features with labels
X = features
y = df['Type'].apply(lambda x: 1 if x == 'Phishing' else 0)  # Convert labels to binary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'phishing_detector_model.pkl')
print("Model saved as 'phishing_detector_model.pkl'")

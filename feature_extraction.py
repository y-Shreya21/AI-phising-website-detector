import pandas as pd
from urllib.parse import urlparse

# Load the dataset
df = pd.read_csv('data\Phishing URLs.csv')

# Function to extract features from the URL
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'has_https': 1 if parsed_url.scheme == 'https' else 0,
        # Add more features as needed
    }
    return features

# Apply feature extraction
features = df['url'].apply(extract_features).apply(pd.Series)

# Combine features with labels
X = features
y = df['Type'].apply(lambda x: 1 if x == 'Phishing' else 0)  # Convert labels to binary

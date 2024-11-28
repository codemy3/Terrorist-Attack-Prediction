import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

# Select relevant columns and preprocess the data
data = data[['country_txt', 'attacktype1_txt', 'iyear', 'nkill']].dropna()

# Add a binary target column: 'attack_occurred' (1 if an attack occurred, else 0)
data['attack_occurred'] = (data['nkill'] > 0).astype(int)

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['country_txt', 'attacktype1_txt'], drop_first=True)

# Define features and target variable
X = data.drop(columns=['iyear', 'nkill', 'attack_occurred'])  # Features
y = data['attack_occurred']  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
classifier.fit(X_train, y_train)

# Save the trained model
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Print training and test set accuracy
print("Training Accuracy:", classifier.score(X_train, y_train))
print("Testing Accuracy:", classifier.score(X_test, y_test))

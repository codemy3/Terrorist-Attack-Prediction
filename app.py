from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained classifier model
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Load the dataset for historical analysis
data = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

# Home route: User inputs country and attack type
@app.route('/')
def index():
    return render_template('index.html')

# Predict route: Processes user input and generates results
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        country = request.form['country']
        attack_type = request.form['attack_type']

        # Filter historical data for the selected country and attack type
        filtered_data = data[(data['country_txt'] == country) & (data['attacktype1_txt'] == attack_type)]
        historical_data = (
            filtered_data.groupby('iyear').size().reset_index(name='attack_count')
        )
        historical_years = historical_data['iyear'].tolist()
        attack_counts = historical_data['attack_count'].tolist()

        # Preprocess the data to zip historical_years and attack_counts
        historical_data = list(zip(historical_years, attack_counts))

        # Prepare input for future prediction (create dummy features for country and attack type)
        future_data = pd.DataFrame({
            f'country_txt_{country}': [1],
            f'attacktype1_txt_{attack_type}': [1]
        })

        # Fill missing columns with 0 (to ensure consistency with training data)
        for col in classifier.feature_names_in_:
            if col not in future_data.columns:
                future_data[col] = 0

        # Ensure the column order matches the training data
        future_data = future_data[classifier.feature_names_in_]

        # Predict likelihood of future attacks
        probabilities = classifier.predict_proba(future_data)

        # Handle single-class predictions and assign default 0% if necessary
        if probabilities.shape[1] > 1:
            likelihood = probabilities[0][1] * 100
        else:
            likelihood = 0  # Default to 0% if no class 1 probability

        # Return the processed data to be displayed on the result page
        return render_template(
            'result.html',
            country=country,
            attack_type=attack_type,
            historical_data=historical_data,
            likelihood=round(likelihood, 2)
        )

    except Exception as e:
        return f"An error occurred: {e}"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

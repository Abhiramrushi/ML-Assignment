import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Define the dataset
data = {
    "Outlook": ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast",
                "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"],
    "Temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild",
                    "cool", "mild", "mild", "mild", "hot", "mild"],
    "Humidity": ["high", "high", "high", "high", "normal", "normal", "normal",
                 "high", "normal", "normal", "normal", "high", "normal", "high"],
    "Windy": ["false", "true", "false", "false", "false", "true", "true", "false",
              "false", "false", "true", "true", "false", "true"],
    "Class": [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],  # 0: No, 1: Yes
}

# Create a DataFrame (Converts the data dictionary into a pandas DataFrame.)
df = pd.DataFrame(data)

# Manual encoding of categorical variables
encoding = {
    "Outlook": {"sunny": 0, "overcast": 1, "rain": 2},
    "Temperature": {"hot": 0, "mild": 1, "cool": 2},
    "Humidity": {"high": 0, "normal": 1},
    "Windy": {"false": 0, "true": 1},
}

# Apply encoding
for column, mapping in encoding.items():
    df[column] = df[column].map(mapping)

# Split into features (X) and target (y)
X = df[["Outlook", "Temperature", "Humidity", "Windy"]]
y = df["Class"]

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Prepare the input for prediction
input_data = {"Outlook": "sunny", "Temperature": "cool", "Humidity": "high", "Windy": "true"}
input_encoded = {
    "Outlook": encoding["Outlook"][input_data["Outlook"]],
    "Temperature": encoding["Temperature"][input_data["Temperature"]],
    "Humidity": encoding["Humidity"][input_data["Humidity"]],
    "Windy": encoding["Windy"][input_data["Windy"]],
}

# Convert the input to a DataFrame with the same feature names
input_df = pd.DataFrame([input_encoded])

# Predict the probability and class
prediction_prob = model.predict_proba(input_df)
prediction = model.predict(input_df)

print(f"Prediction Probability: {prediction_prob}")
print(f"Prediction (0: No, 1: Yes): {prediction[0]}")
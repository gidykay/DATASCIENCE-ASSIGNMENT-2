import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. Load the dataset
# Ensure Road.csv is in the same directory as this script.
df = pd.read_csv("Road.csv")

# Display basic info to verify correct loading
print("Data preview:")
print(df.head())
print("\nData info:")
print(df.info())

# 2. Preprocess the data
# If there is an "Additional_Info" column (from our CSV fix), we drop it as it's non-numeric.
if "Additional_Info" in df.columns:
    df = df.drop(columns=["Additional_Info"])

# Convert categorical variables to dummy/indicator variables.
# In this example, we assume "Weather_Condition" and "Road_Condition" are categorical.
df_encoded = pd.get_dummies(df, columns=["Weather_Condition", "Road_Condition"], drop_first=True)

# 3. Split the data into features and target
# Assume "Accident_Severity" is the target variable.
X = df_encoded.drop("Accident_Severity", axis=1)
y = df_encoded["Accident_Severity"]

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation Metrics:")
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)

# 5. Save the trained model to a pickle file
with open("road_accident_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\nModel saved as 'road_accident_model.pkl'")

# 6. Generate and save plots

# Plot 1: Actual vs Predicted Accident Severity
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.xlabel("Actual Accident Severity")
plt.ylabel("Predicted Accident Severity")
plt.title("Actual vs Predict

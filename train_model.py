import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 1. Load the dataset
data = pd.read_csv('company_ratios.csv')

# 2. Split into features (X) and target (y)
# Assumes all columns except 'Company', 'Year', and 'Performance_Score' are ratio features
X = data.drop(columns=['Company', 'Year', 'Performance_Score'])
y = data['Performance_Score']

# 3. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Random Forest model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 6. Show which ratios matter most
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False))

# 7. Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")

# 8. Example: predict on a new company's ratios
# Replace these values with a real company's ratios in the same column order as X
sample = X.iloc[[0]]  # using first row as an example
sample_prediction = model.predict(sample)
print(f"\nSample prediction: {sample_prediction[0]:.2f}")

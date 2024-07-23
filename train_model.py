import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle

# Load your data
data = pd.read_csv('E:/data_science_projects/Nigeria Economy/dataset/1960_onwards.csv')

# Independent variables
independent_vars = [
    "Year",
    "Consumer price index (2010 = 100)",
    "GDP (current LCU)",
    "Inflation, GDP deflator (annual %)",
    "Official exchange rate (LCU per US$, period average)",
    "Total reserves (includes gold, current US$)",
    "Population, total",
    "Population ages 15-64 (% of total population)",
    "Money Supply M3",
    "Base Money",
    "Currency in Circulation",
    "Bank Reserves",
    "Currency Outside Banks",
    "Quasi Money",
    "Other Assets Net",
    "CBN Bills",
    "Special Intervention Reserves",
    "GDPBillions of US $",
    "Per CapitaUS $",
    "Petrol Price (Naira)"
]

# Target variable
target_var = "GDP per capita (current US$)"

# Define features and target
X = data[independent_vars]
y = data[target_var]

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = Lasso(alpha=0.1)  # Adjust alpha as needed
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Save the model and scaler
with open('lasso_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

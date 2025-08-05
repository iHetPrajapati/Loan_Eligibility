# In main.py or a separate script
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("credit.csv")
print(df.columns)



# Preprocessing (same as before)
df = df.dropna()
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Features and target
X = df.drop(['Loan_ID', 'Loan_Approved'], axis=1)
y = df['Loan_Approved'].map({'Y': 1, 'N': 0})



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
import os
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/model.pkl")

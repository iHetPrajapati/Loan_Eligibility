from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    # Replace '3+' in Dependents and convert to int
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    le = LabelEncoder()
    categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Approved']
    for col in categorical:
        df[col] = le.fit_transform(df[col])
    return df

# utils.py
import pandas as pd

def preprocess_input(input_df):
    # Copy input
    df = input_df.copy()

    # Convert categorical columns (same as during training)
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3}
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    return df


def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

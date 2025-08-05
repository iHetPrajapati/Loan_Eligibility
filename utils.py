from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    # Replace '3+' in Dependents and convert to int
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    le = LabelEncoder()
    categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Approved']
    for col in categorical:
        df[col] = le.fit_transform(df[col])
    return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Preprocessing 
# Separate dataset into categorical and numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Imputation for categorical data (using most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Ordinal encoding
ordinal_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 
                   'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 
                   'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']

ordinal_mapping = [
    ['Low', 'Medium', 'High'],          # Parental_Involvement
    ['Low', 'Medium', 'High'],          # Access_to_Resources
    ['No', 'Yes'],                      # Extracurricular_Activities
    ['Low', 'Medium', 'High'],          # Motivation_Level
    ['No', 'Yes'],                      # Internet_Access
    ['Low', 'Medium', 'High'],          # Family_Income
    ['Low', 'Medium', 'High'],          # Teacher_Quality
    ['Public', 'Private'],              # School_Type
    ['Negative', 'Neutral', 'Positive'],# Peer_Influence
    ['No', 'Yes'],                      # Learning_Disabilities
    ['High School', 'College', 'Postgraduate'], # Parental_Education_Level
    ['Near', 'Moderate', 'Far'],        # Distance_from_Home
    ['Male', 'Female']                  # Gender
]

# Applying ordinal encoding
ordinal_encoder = OrdinalEncoder(categories=ordinal_mapping)
df[ordinal_columns] = ordinal_encoder.fit_transform(df[ordinal_columns])

#  Split the dataset into training and testing sets
X = df.drop("Exam_Score", axis=1)  # Features
y = df['Exam_Score']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using Random Forest Regressor 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model prediction and evaluation
y_pred = rf_model.predict(X_test)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Feature importance plot
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("RF Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

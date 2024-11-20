import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('StudentPerformanceFactors.csv')

# Split the data into features (X) and target (y)
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
print("Dataset types")
df.shape
df.info()

#for column in df.columns:
#    plt.figure(figsize=(10, 5))
#    sns.histplot(df[column], bins=30, kde=True)
#    plt.title(f'Distribution of {column}')
#    plt.xlabel(column)
#    plt.ylabel('Frequency')
#    plt.show()

# Calculate number of Null
print("\nNumber of Null")
print(df.isnull().sum())

# Calculate number of Null in  %
print("\nNumber of Null in %")
print(df.isnull().sum() / df.shape[0] * 100)

# Separate dataset into categorical and numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Imputation for categorical data (using most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Calculate number of Null
print("\nNumber of Null")
print(df[categorical_cols].isnull().sum())


# Ordinal categories for encoding
ordinal_columns = ['Parental_Involvement', 'Access_to_Resources','Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 
                   'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
ordinal_mapping = [
    ['Low', 'Medium', 'High'],          # Parental_Involvement
    ['Low', 'Medium', 'High'],          # Access_to_Resources
    ['No', 'Yes'],                      # Extracurricular_Activities
    ['Low', 'Medium', 'High'],          # Motivation_Level
    ['No', 'Yes'],                      # Internet_Access
    ['Low', 'Medium', 'High'],          # Family_Income
    ['Low', 'Medium', 'High'],          # Teacher_Quality
    ['Public', 'Private'],              # 'School_Type'
    ['Negative', 'Neutral', 'Positive'] ,       # 'Peer_Influence' 
    ['No', 'Yes'],                              # 'Learning_Disabilities'
    ['High School', 'College', 'Postgraduate'], # 'Parental_Education_Level'
    ['Near', 'Moderate', 'Far'],                # 'Distance_from_Home'
    ['Male', 'Female']                          # 'Gender'
]

# Applying ordinal encoding
ordinal_encoder = OrdinalEncoder(categories=ordinal_mapping)
df[ordinal_columns] = ordinal_encoder.fit_transform(df[ordinal_columns])

print("After Ordinal Encoding:\n", df)

encoded_columns = df.select_dtypes(include=['float64', 'int64','int32']).columns
encoded_columns_list = df[encoded_columns]

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(encoded_columns_list.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
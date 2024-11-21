import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('alzheimers_disease_data.csv')

print("Dataset types")
df.shape
df.info()

for column in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Calculate number of Null
print("\nNumber of Null")
print(df.isnull().sum())

# Calculate number of Null in  %
print("\nNumber of Null in %")
print(df.isnull().sum() / df.shape[0] * 100)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop('DoctorInCharge', axis=1).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
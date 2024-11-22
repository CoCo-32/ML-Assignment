import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('alzheimers_disease_data.csv')

# Preprocessing 
# Split the dataset into training and testing sets
X = df.drop(['DoctorInCharge', 'Diagnosis', 'PatientID'], axis=1)  # Features
y = df['Diagnosis']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define the parameter grid for RandomizedSearchCV
xgb_param_grid = {
    'learning_rate': [0.05, 0.01, 0.1, 0.2, 0.3],  
    'max_depth': [3, 5, 7, 9, 12],                 # Depth of each tree
    'n_estimators': [50, 100, 150, 200],           # Number of boosting rounds (trees)
    'subsample': [0.7, 0.8, 0.9, 1.0],             # Fraction of samples used for each tree
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],      # Fraction of features used for each tree
    'gamma': [0, 0.1, 0.3, 0.5],                   # Minimum loss reduction required to make a further partition
    'scale_pos_weight': [1, 2, 3]                  # Controls the balance of positive and negative weights
}

# Initialize the RandomizedSearchCV
random_search_xgb = RandomizedSearchCV(estimator=xgb_model, 
                                   param_distributions=xgb_param_grid, 
                                   n_iter=100,          # Number of random combinations to try
                                   cv=8,                
                                   n_jobs=-1,           # Use all available CPU cores
                                   verbose=1,
                                   scoring='accuracy',  # Metric to optimize for
                                   random_state=42)

# Perform the random search on the training data
random_search_xgb.fit(X_train, y_train)

# Get the best hyperparameters found by RandomizedSearchCV
best_params_random = random_search_xgb.best_params_
print("Best hyperparameters found using RandomizedSearchCV:", best_params_random)

# Train a model using the best parameters found
best_model = random_search_xgb.best_estimator_

# Make predictions with the best model
y_pred_best = best_model.predict(X_test)

# Evaluate the best model's performance
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Best Model Accuracy: {accuracy_best:.2f}")
print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred_best))

# Plot confusion matrix for the best model
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Best Model')
plt.show()

# Feature importances plot for the best model
importances_best = best_model.feature_importances_
indices_best = np.argsort(importances_best)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances of Best XGBoost Model")
plt.bar(range(X_train.shape[1]), importances_best[indices_best], align='center')
plt.xticks(range(X_train.shape[1]), X.columns[indices_best], rotation=90)
plt.tight_layout()
plt.show()

# Compute and plot the ROC curve for the best model
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

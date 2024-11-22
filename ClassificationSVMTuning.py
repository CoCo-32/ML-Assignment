import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Load the dataset
df = pd.read_csv('alzheimers_disease_data.csv')

# Preprocessing 
# Splitting the dataset into features (X) and target (y)
X = df.drop(['DoctorInCharge', 'Diagnosis', 'PatientID'], axis=1)  # Features
y = df['Diagnosis']  # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm_model = SVC(random_state=42)

# Define the parameter grid for RandomizedSearchCV
svm_param_grid = {
    'C': uniform(1, 1000),                                # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 1],                 # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly', ],                # Type of kernel
    'degree': randint(2, 6),                              # Degree for 'poly' kernel
    'tol': uniform(1e-5, 1e-2),                           # Tolerance for stopping criterion
}

# Initialize RandomizedSearchCV
random_search_svm = RandomizedSearchCV(estimator=svm_model, 
                                   param_distributions=svm_param_grid, 
                                   n_iter=100,           # Number of random combinations to try
                                   cv=5,                 
                                   n_jobs=-1,            # Use all available CPU cores
                                   verbose=1,
                                   scoring='accuracy',   # Metric to optimize for
                                   random_state=42)

# Perform the random search on the training data
random_search_svm.fit(X_train, y_train)

# Get the best hyperparameters found by RandomizedSearchCV
best_params_random = random_search_svm.best_params_
print("Best hyperparameters found using RandomizedSearchCV:", best_params_random)

# Train a model using the best parameters found
best_model = random_search_svm.best_estimator_

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

# Feature importances plot for the best model (if available)
# SVM does not directly provide feature importances, but you can examine the coefficients for linear kernels
if best_model.kernel == 'linear':
    coef = best_model.coef_.toarray().flatten()
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, coef)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Feature Coefficients for Linear SVM')
    plt.show()

# Optionally: Compute and plot the ROC curve for the best model
fpr, tpr, thresholds = roc_curve(y_test, best_model.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

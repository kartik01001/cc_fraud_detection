import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('updated_dataset.csv')

# Drop TransactionID (non-relevant to prediction)
df.drop(['TransactionID'], axis=1, inplace=True)

# Handle categorical data with LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns
categorical_cols = ['MerchantID', 'MerchantCategoryCode', 'CardholderID', 'CardholderLocation', 'CardType', 'IssuerBank', 'DeviceType', 'IPAddress', 'IsHighRiskMerchant']

# Apply label encoding to categorical columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features and target
X = df.drop('IsFraud', axis=1)  # Features
y = df['IsFraud']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the numerical columns
scaler = StandardScaler()
numerical_cols = ['TimeBetweenTransactions', 'GeolocationDistance']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the training dataset
y_train_pred = model.predict(X_train)

# Threshold predictions to get binary results for training set
y_train_pred_thresholded = [1 if pred >= 0.5 else 0 for pred in y_train_pred]

# Calculate accuracy for training set
accuracy_train = accuracy_score(y_train, y_train_pred_thresholded)

# Print evaluation metrics for training set
print(f"Training Accuracy: {accuracy_train * 100:.2f}%")

# Plotting the confusion matrix for training set
cm_train = confusion_matrix(y_train, y_train_pred_thresholded)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Training Set)')
plt.show()


# Make predictions on the test dataset
y_pred = model.predict(X_test)

# Threshold predictions to get binary results for test set
y_pred_thresholded = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Calculate accuracy for test set
accuracy_test = accuracy_score(y_test, y_pred_thresholded)

# Print evaluation metrics for test set
print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

# Plotting the confusion matrix for test set
cm_test = confusion_matrix(y_test, y_pred_thresholded)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Set)')
plt.show()




custom_input = pd.DataFrame({
    'MerchantID': ['M145'],  # Example value
    'MerchantCategoryCode': ['5941'],
    'CardholderID': ['C123'],
    'CardholderLocation': ['US'],
    'CardType': ['Credit'],
    'IssuerBank': ['BankA'],
    'CardExpirationDate': ['6112026'],
    'DeviceType': ['Desktop'],
    'IPAddress': ['192.168.0.45'],
    'IsHighRiskMerchant': [0],  # Not a high-risk merchant
    'TimeBetweenTransactions': [184],
    'GeolocationDistance': [582.83]
})

# Apply the same Label Encoding to categorical data
for col in categorical_cols:
    custom_input[col] = label_encoder.fit_transform(custom_input[col])

# Apply the same scaling to numerical data
custom_input[numerical_cols] = scaler.transform(custom_input[numerical_cols])

# Make prediction for the custom input
custom_prediction = model.predict(custom_input)

# Output the prediction (0 = Not Fraud, 1 = Fraud)
print(f"Custom Input Prediction: {'Fraud' if custom_prediction[0] == 1 else 'Not Fraud'}")
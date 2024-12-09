
# Telecom Churn Prediction

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset
data_path = 'telecom_churn_data.csv'
telecom_data = pd.read_csv(data_path)

# Data preprocessing
# Step 1: Calculate average recharge amount for good phase (months 6 and 7)
telecom_data['avg_rech_amt_good_phase'] = telecom_data[['total_rech_amt_6', 'total_rech_amt_7']].mean(axis=1)

# Step 2: Define high-value customers (70th percentile threshold)
threshold_70_percentile = telecom_data['avg_rech_amt_good_phase'].quantile(0.7)
high_value_customers = telecom_data[telecom_data['avg_rech_amt_good_phase'] >= threshold_70_percentile]

# Step 3: Tag churners based on usage criteria for month 9
churn_criteria = (
    (high_value_customers['total_ic_mou_9'] == 0) &
    (high_value_customers['total_og_mou_9'] == 0) &
    (high_value_customers['vol_2g_mb_9'] == 0) &
    (high_value_customers['vol_3g_mb_9'] == 0)
)
high_value_customers['churn'] = churn_criteria.astype(int)

# Step 4: Remove churn-phase data
columns_to_drop = [col for col in high_value_customers.columns if col.endswith('_9')]
high_value_customers_cleaned = high_value_customers.drop(columns=columns_to_drop, axis=1)

# Step 5: Prepare data for modeling
X = high_value_customers_cleaned.drop(columns=['churn', 'mobile_number'], axis=1)
y = high_value_customers_cleaned['churn']
X.fillna(X.median(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Train a Logistic Regression model
logistic_model = LogisticRegression(class_weight='balanced', random_state=42)
logistic_model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = logistic_model.predict(X_test)
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': high_value_customers_cleaned.drop(columns=['churn', 'mobile_number']).columns,
    'Importance': logistic_model.coef_[0]
}).sort_values(by='Importance', ascending=False)
print("Top Features Influencing Churn:")
print(feature_importance.head(10))

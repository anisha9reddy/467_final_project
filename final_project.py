import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# PART 1 - LOADING DATA and DATA CLEANING

cancer_df = pd.read_csv("esophageal_cancer.csv")
invalid_data = ['Blank(s)', '999', 'Unknown']
cancer_df = cancer_df.replace(invalid_data, pd.NA)

# delete rows for which 6th-T to 8th-M are all blank
columns_to_check = ['6th-T', '6th-N', '6th-M', '7th-T', '7th-N', '7th-M', '8th-T', '8th-N', '8th-M']
cancer_df = cancer_df.dropna(subset=columns_to_check, how='all')
# delete rows with no race info
cancer_df = cancer_df.dropna(subset=['race'])
# delete rows with no cancer grade info
cancer_df = cancer_df.dropna(subset=['grade'])
# delete unnamed column
cancer_df = cancer_df.drop(columns=[cancer_df.columns[5]])
# delete rows where tumor size is NA
cancer_df = cancer_df.dropna(subset=['tumor size'])
# delete rows without survival months info
cancer_df = cancer_df.dropna(subset=['survival months'])
# delete columns with no values or irrelevant/less relevant values
cancer_df = cancer_df.drop(columns=['7th-T', '7th-N', '7th-M', '8th-T', '8th-N', '8th-M'])
# delete people who didn't die from cancer
cancer_df = cancer_df[cancer_df['cause-specific death'] == 'Dead (attributable to this cancer dx)']
# delete columns with irrelevant/less relevant features
cancer_df = cancer_df.drop(columns=['histology', 'sequence number', 'year of diagnosis', 'othor cause of death', 'first malignant primary indicator', 'primary site', 'primary site.1', 'surg prim site'])
# patients who died from cancer...want a model that predicts how many months patients have left
dead_cancer_df = cancer_df[cancer_df['vital status'] == 'Dead']
# no longer need vital status or death info bc we know everyone died from cancer
dead_cancer_df = dead_cancer_df.drop(columns=['vital status', 'cause-specific death'])
print(len(dead_cancer_df))

# PART 2 - DATA FORMATTING

# converting data to numerical data
dead_cancer_df['age'] = dead_cancer_df['age'].replace('85+ years', '85')
dead_cancer_df['age'] = dead_cancer_df['age'].str.extract('(\d+)').astype(int)
dead_cancer_df['survival months'] = dead_cancer_df['survival months'].astype(int)
# one hot encoding categorial variables
categorical_features = ['sex', 'race', 'grade', '6th-T', '6th-N', '6th-M']
dead_cancer_df = pd.get_dummies(dead_cancer_df, columns=categorical_features, drop_first=True)



# PART 3 - INITIAL DATASET VISUALIZATIONS

# Plot distribution of survival months
plt.figure(figsize=(10, 6))
sns.histplot(data=dead_cancer_df, x='survival months', kde=True, bins=50)
plt.title("Distribution of Patients by Survival Months")
plt.xlabel("Survival Months")
plt.ylabel("Number of Patients")
plt.show()

# Count patients based on survival months
lived_longer = dead_cancer_df[dead_cancer_df['survival months'] > 6].shape[0]
died_immediately = dead_cancer_df[dead_cancer_df['survival months'] <= 6].shape[0]
print(f"Patients who survived more than 6 months: {lived_longer}")
print(f"Patients who survived 6 months or less: {died_immediately}")

plt.figure(figsize=(10, 6))
sns.histplot(data=dead_cancer_df[dead_cancer_df['survival months'] <= 24], x='survival months', kde=True, bins=20)
plt.title("Distribution of Patients by Survival Months (0-24 Months)")
plt.xlabel("Survival Months")
plt.ylabel("Number of Patients")
plt.show()


# PART 4 - BASELINE MODEL DEVELOPMENT (Logistic Regression)

# Set up binary classification target (using 1 year or less to indicate high risk)
survival_threshold = 6
dead_cancer_df['high_risk'] = (dead_cancer_df['survival months'] <= survival_threshold).astype(int)

# Defining features and target for classification
X = dead_cancer_df.drop(columns=['survival months', 'high_risk'])
y = dead_cancer_df['high_risk']

# Split the data and scaling the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# using a multivariate logistic regression model
multi_model = LogisticRegression(max_iter=5000, random_state=42)
multi_model.fit(X_train_scaled, y_train)

# make predictions and evaluate results
y_pred = multi_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Multivariate Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# BASELINE MODEL EVALUTION

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Initialize a dictionary to store performance metrics for each race
race_metrics = {}

# Get the race columns in the one-hot encoded dataset (those starting with 'race_')
race_columns = [col for col in X_test.columns if col.startswith('race_')]

# Iterate through each race column and evaluate the model
for race_col in race_columns:
    # Filter test data for each racial group
    race_test_indices = X_test[X_test[race_col] == 1].index
    X_test_race = X_test.loc[race_test_indices]  # Get original test data for this race group
    X_test_race_scaled = scaler.transform(X_test_race)  # Scale the subset
    y_test_race = y_test.loc[race_test_indices]  # Corresponding labels for this race group

    # Predict and evaluate for the current racial group
    y_pred_race = multi_model.predict(X_test_race_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test_race, y_pred_race)
    class_report = classification_report(y_test_race, y_pred_race, output_dict=True)

    # Store metrics in dictionary
    race_metrics[race_col] = {
        "Accuracy": accuracy,
        "Precision": class_report["1"]["precision"],
        "Recall": class_report["1"]["recall"],
        "F1-Score": class_report["1"]["f1-score"]
    }

# Baseline model performance on each racial group
for race, metrics in race_metrics.items():
    print(f"Performance for {race}:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print("\n")


# PART 5 - HANDLING IMBALANCE WITH SMOTE

# Using SMOTE to improve accuracy on black patients
# Identify Black patients in the training set
black_train_indices = X_train[X_train['race_Black'] == 1].index
X_train_black = X_train.loc[black_train_indices]
y_train_black = y_train.loc[black_train_indices]

# Apply SMOTE only to the Black patient subgroup
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_black_smote, y_train_black_smote = smote.fit_resample(X_train_black, y_train_black)

# Recombine the data with the rest of the training set
X_train_balanced = pd.concat([X_train.drop(black_train_indices), X_train_black_smote])
y_train_balanced = pd.concat([y_train.drop(black_train_indices), y_train_black_smote])

# Scale the balanced training data
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model on the balanced dataset
multi_model_smote = LogisticRegression(max_iter=5000, random_state=42)
multi_model_smote.fit(X_train_balanced_scaled, y_train_balanced)

# Make predictions and evaluate the results on the test set
y_pred_smote = multi_model_smote.predict(X_test_scaled)
accuracy_smote = accuracy_score(y_test, y_pred_smote)
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
class_report_smote = classification_report(y_test, y_pred_smote)

print("SMOTE-Adjusted Logistic Regression Results:")
print(f"Accuracy: {accuracy_smote:.4f}")
print("Confusion Matrix:")
print(conf_matrix_smote)

# Confusion Matrix Heatmap for SMOTE Model
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_smote, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix (SMOTE Adjusted)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Check model performance on each racial group after SMOTE adjustment
race_metrics_smote = {}

for race_col in race_columns:
    # Filter test data for each racial group
    race_test_indices = X_test[X_test[race_col] == 1].index
    X_test_race = X_test.loc[race_test_indices]  # Get original test data for this race group
    X_test_race_scaled = scaler.transform(X_test_race)  # Scale the subset
    y_test_race = y_test.loc[race_test_indices]  # Corresponding labels for this race group

    # Predict and evaluate for the current racial group
    y_pred_race_smote = multi_model_smote.predict(X_test_race_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test_race, y_pred_race_smote)
    class_report = classification_report(y_test_race, y_pred_race_smote, output_dict=True)

    # Store metrics in dictionary
    race_metrics_smote[race_col] = {
        "Accuracy": accuracy,
        "Precision": class_report["1"]["precision"],
        "Recall": class_report["1"]["recall"],
        "F1-Score": class_report["1"]["f1-score"]
    }

# Display the metrics for each racial group after SMOTE adjustment
for race, metrics in race_metrics_smote.items():
    print(f"Performance for {race} (SMOTE Adjusted):")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print("\n")


# PART 5 - HANDLE IMBALANCE W/ Random Forest with SMOTEENN

from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN  
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# Apply SMOTEENN to balance the training data
smote_enn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Random Forest Results with SMOTEENN:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Get precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Analyze performance by racial subgroup
race_metrics_rf = {}

for race_col in race_columns:
    race_test_indices = X_test[X_test[race_col] == 1].index
    X_test_race = X_test.loc[race_test_indices]
    y_test_race = y_test.loc[race_test_indices]
    y_pred_race = rf_model.predict(X_test_race)
    y_proba_race = rf_model.predict_proba(X_test_race)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test_race, y_pred_race)
    roc_auc = roc_auc_score(y_test_race, y_proba_race)
    class_report = classification_report(y_test_race, y_pred_race, output_dict=True)

    # Store metrics
    race_metrics_rf[race_col] = {
        "Accuracy": accuracy,
        "ROC AUC": roc_auc,
        "Precision": class_report["1"]["precision"],
        "Recall": class_report["1"]["recall"],
        "F1-Score": class_report["1"]["f1-score"]
    }

# Display the metrics for each racial group
for race, metrics in race_metrics_rf.items():
    print(f"Performance for {race} with Random Forest:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"ROC AUC: {metrics['ROC AUC']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print("\n")


# PART 6 - TRAINING A MODEL FOR JUST ONE RACIAL SUBGROUP (BLACK PATIENTS)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Step 1: Filter the dataset for Black patients only
black_df = dead_cancer_df[dead_cancer_df['race_Black'] == 1]

# Define features and target for Black patients
X_black = black_df.drop(columns=['survival months', 'high_risk'])
y_black = black_df['high_risk']

# Step 2: Split the data for Black patients and apply scaling
X_train_black, X_test_black, y_train_black, y_test_black = train_test_split(X_black, y_black, test_size=0.3, random_state=42)

# Apply SMOTE if needed (only if there’s class imbalance in the training set for Black patients)
smote = SMOTE(random_state=42)
X_train_black_smote, y_train_black_smote = smote.fit_resample(X_train_black, y_train_black)

# Scale the data
scaler = StandardScaler()
X_train_black_scaled = scaler.fit_transform(X_train_black_smote)
X_test_black_scaled = scaler.transform(X_test_black)

# Step 3: Train the model
model_black = LogisticRegression(max_iter=5000, random_state=42)
model_black.fit(X_train_black_scaled, y_train_black_smote)

# Step 4: Evaluate the model on Black patients' test set
y_pred_black = model_black.predict(X_test_black_scaled)
accuracy_black = accuracy_score(y_test_black, y_pred_black)
conf_matrix_black = confusion_matrix(y_test_black, y_pred_black)
class_report_black = classification_report(y_test_black, y_pred_black)

print("Logistic Regression Results for Black Patients Only:")
print(f"Accuracy: {accuracy_black:.4f}")
print("Confusion Matrix:")
print(conf_matrix_black)
print("Classification Report:")
print(class_report_black)


# PART 7 - GENERATING VISUALS FOR FINAL REPORT
# Count the occurrences of each race
race_counts = cancer_df['race'].value_counts()
race_counts.index = race_counts.index.str.replace('American Indian/Alaska Native', 'American Indian')
race_counts.index = race_counts.index.str.replace('Asian or Pacific Islander', 'Asian')

# Plot the bar chart
plt.figure(figsize=(10, 8))
ax = race_counts.plot(kind='bar', title='Racial Diversity in Esophageal Cancer Dataset')
plt.xlabel('Race')
plt.ylabel('Count')

# Rotate x-axis labels
plt.xticks(rotation=0)

# Add exact counts on top of each bar
for i, count in enumerate(race_counts):
    ax.text(i, count + 0.05 * count, str(count), ha='center', va='bottom')

plt.show()


# PART 8 - EXPERIMENTS/EVALUATION + GENERATING VISUALS (CONT.)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to calculate metrics for each model
def get_model_metrics(y_true, y_pred, y_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics

# Calculate metrics for Baseline Model
y_pred_baseline = multi_model.predict(X_test_scaled)
y_proba_baseline = multi_model.predict_proba(X_test_scaled)[:, 1]
baseline_metrics = get_model_metrics(y_test, y_pred_baseline, y_proba_baseline)

# Calculate metrics for SMOTE-Enhanced Model
y_pred_smote = multi_model_smote.predict(X_test_scaled)
y_proba_smote = multi_model_smote.predict_proba(X_test_scaled)[:, 1]
smote_metrics = get_model_metrics(y_test, y_pred_smote, y_proba_smote)

# Calculate metrics for Random Forest Model with SMOTEENN
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
rf_metrics = get_model_metrics(y_test, y_pred_rf, y_proba_rf)

# # Organizing performance data for plotting
# Initialize a dictionary to store performance metrics for White and Black groups
white_black_metrics = {
    "Race": ["White", "Black"],
    "Baseline Accuracy": [],
    "Baseline F1-Score": [],
    "SMOTE Accuracy": [],
    "SMOTE F1-Score": [],
    "Random Forest Accuracy": [],
    "Random Forest F1-Score": []
}

# For Baseline Model
white_black_metrics["Baseline Accuracy"].append(race_metrics["race_White"]["Accuracy"])
white_black_metrics["Baseline F1-Score"].append(race_metrics["race_White"]["F1-Score"])
white_black_metrics["Baseline Accuracy"].append(race_metrics["race_Black"]["Accuracy"])
white_black_metrics["Baseline F1-Score"].append(race_metrics["race_Black"]["F1-Score"])

# For SMOTE-Enhanced Model
white_black_metrics["SMOTE Accuracy"].append(race_metrics_smote["race_White"]["Accuracy"])
white_black_metrics["SMOTE F1-Score"].append(race_metrics_smote["race_White"]["F1-Score"])
white_black_metrics["SMOTE Accuracy"].append(race_metrics_smote["race_Black"]["Accuracy"])
white_black_metrics["SMOTE F1-Score"].append(race_metrics_smote["race_Black"]["F1-Score"])

# For Random Forest with SMOTEENN Model
white_black_metrics["Random Forest Accuracy"].append(race_metrics_rf["race_White"]["Accuracy"])
white_black_metrics["Random Forest F1-Score"].append(race_metrics_rf["race_White"]["F1-Score"])
white_black_metrics["Random Forest Accuracy"].append(race_metrics_rf["race_Black"]["Accuracy"])
white_black_metrics["Random Forest F1-Score"].append(race_metrics_rf["race_Black"]["F1-Score"])

# Round to 4 decimal points
accuracy_metrics = {
    "Race": white_black_metrics["Race"],
    "Baseline Accuracy": [round(val, 4) for val in white_black_metrics["Baseline Accuracy"]],
    "SMOTE Accuracy": [round(val, 4) for val in white_black_metrics["SMOTE Accuracy"]],
    "Random Forest Accuracy": [round(val, 4) for val in white_black_metrics["Random Forest Accuracy"]]
}

f1_score_metrics = {
    "Race": white_black_metrics["Race"],
    "Baseline F1-Score": [round(val, 4) for val in white_black_metrics["Baseline F1-Score"]],
    "SMOTE F1-Score": [round(val, 4) for val in white_black_metrics["SMOTE F1-Score"]],
    "Random Forest F1-Score": [round(val, 4) for val in white_black_metrics["Random Forest F1-Score"]]
}

# Convert to DataFrame
accuracy_df = pd.DataFrame(accuracy_metrics)
f1_score_df = pd.DataFrame(f1_score_metrics)

def display_metrics_table(df, title):
    fig, ax = plt.subplots(figsize=(10, 2))  
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(title)
    plt.show()

# Display each table
display_metrics_table(accuracy_df, "Accuracy Results Across Models Based on Race")
display_metrics_table(f1_score_df, "F1-Score Results Across Model Based on Race")


# PART 9 - INCORPORATING FAIRNESS METRICS INTO VARIOUS MODELS
def evaluate_fairness(y_true, y_pred, sensitive_features):
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return dp_diff, eo_diff

# FAIRNESS-AWARE BASELINE LOGISTIC REGRESSION
baseline_fair_model = ExponentiatedGradient(
    LogisticRegression(max_iter=5000, random_state=42),
    constraints=DemographicParity()
)

baseline_fair_model.fit(
    X_train_scaled,
    y_train,
    sensitive_features=X_train[['race_Black', 'race_White']].idxmax(axis=1)
)
y_pred_baseline_fair = baseline_fair_model.predict(X_test_scaled)
dp_diff_baseline, eo_diff_baseline = evaluate_fairness(
    y_test,
    y_pred_baseline_fair,
    sensitive_features=X_test[['race_Black', 'race_White']].idxmax(axis=1)
)
print("Fairness-Aware Baseline Logistic Regression Results:")
print(f"Demographic Parity Difference: {dp_diff_baseline:.4f}")
print(f"Equalized Odds Difference: {eo_diff_baseline:.4f}")

# FAIRNESS-AWARE SMOTE-ENHANCED LOGISTIC REGRESSION
smote_fair_model = ExponentiatedGradient(
    LogisticRegression(max_iter=5000, random_state=42),
    constraints=DemographicParity()
)

X_train_balanced = pd.concat([X_train.drop(black_train_indices), X_train_black_smote])
y_train_balanced = pd.concat([y_train.drop(black_train_indices), y_train_black_smote])

X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

sensitive_features_balanced = X_train_balanced[['race_Black', 'race_White']]
sensitive_features_test = X_test[['race_Black', 'race_White']]

smote_fair_model.fit(
    X_train_balanced_scaled,
    y_train_balanced,
    sensitive_features=sensitive_features_balanced.idxmax(axis=1)
)

y_pred_smote_fair = smote_fair_model.predict(X_test_scaled)
dp_diff_smote, eo_diff_smote = evaluate_fairness(
    y_test,
    y_pred_smote_fair,
    sensitive_features=sensitive_features_test
)
print("Fairness-Aware SMOTE-Enhanced Logistic Regression Results:")
print(f"Demographic Parity Difference: {dp_diff_smote:.4f}")
print(f"Equalized Odds Difference: {eo_diff_smote:.4f}")

# FAIRNESS-AWARE RANDOM FOREST WITH SMOTEENN
rf_fair_model = ExponentiatedGradient(
    RandomForestClassifier(n_estimators=100, random_state=42),
    constraints=DemographicParity()
)

rf_fair_model.fit(
    X_train_balanced,
    y_train_balanced,
    sensitive_features=sensitive_features_balanced.idxmax(axis=1)
)
y_pred_rf_fair = rf_fair_model.predict(X_test)
dp_diff_rf, eo_diff_rf = evaluate_fairness(
    y_test,
    y_pred_rf_fair,
    sensitive_features=sensitive_features_test
)
print("Fairness-Aware Random Forest Results:")
print(f"Demographic Parity Difference: {dp_diff_rf:.4f}")
print(f"Equalized Odds Difference: {eo_diff_rf:.4f}")

# PART 10 - ADVERSARIAL DEBIASING

# Prepare data
sensitive_features = X[['race_Black', 'race_White']].idxmax(axis=1)  # Assume 'race_Black' and 'race_White' are one-hot encoded
sensitive_features = sensitive_features.map({'race_Black': 1, 'race_White': 0})  # Binary encoding for adversary
X_train_adv, X_test_adv, y_train_adv, y_test_adv, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_features, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_adv_scaled = scaler.fit_transform(X_train_adv)
X_test_adv_scaled = scaler.transform(X_test_adv)

# Define the predictor model
def build_predictor(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    return Model(inputs, predictions)

# Define the adversary model
def build_adversary(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    return Model(inputs, predictions)

# Build and compile models
input_dim = X_train_adv_scaled.shape[1]
predictor = build_predictor(input_dim)
adversary = build_adversary(1)  # Adversary takes output of predictor as input

predictor.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
adversary.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train predictor and adversary iteratively
batch_size = 32
epochs = 20
adversary_weight = 0.1 

for epoch in range(epochs):
    # Train predictor
    predictor.fit(X_train_adv_scaled, y_train_adv, epochs=1, batch_size=batch_size, verbose=1, validation_split=0.2)
    
    # Generate predictions for adversary training
    predictions = predictor.predict(X_train_adv_scaled)
    
    # Train adversary
    adversary.fit(predictions, sensitive_train, epochs=1, batch_size=batch_size, verbose=1, validation_split=0.2)
    
    # Update predictor to fool adversary
    with tf.GradientTape() as tape:
        predictor_predictions = predictor(tf.convert_to_tensor(X_train_adv_scaled, dtype=tf.float32))
        y_train_adv_reshaped = tf.reshape(y_train_adv, (-1, 1))
        predictor_loss = tf.keras.losses.binary_crossentropy(y_train_adv_reshaped, predictor_predictions)
        
        adversary_predictions = adversary(predictor_predictions)
        sensitive_train_reshaped = tf.reshape(sensitive_train, (-1, 1))
        adversary_loss = tf.keras.losses.binary_crossentropy(sensitive_train_reshaped, adversary_predictions)
        
        combined_loss = predictor_loss - adversary_weight * adversary_loss  # Maximize adversary loss
        
    grads = tape.gradient(combined_loss, predictor.trainable_variables)
    predictor.optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

# Evaluate the predictor on the test set
y_pred_adv = (predictor.predict(X_test_adv_scaled) > 0.5).astype(int)
accuracy_adv = accuracy_score(y_test_adv, y_pred_adv)
conf_matrix_adv = confusion_matrix(y_test_adv, y_pred_adv)
class_report_adv = classification_report(y_test_adv, y_pred_adv)

print("Adversarial Debiasing Predictor Results:")
print(f"Accuracy: {accuracy_adv:.4f}")
print("Confusion Matrix:")
print(conf_matrix_adv)
print("Classification Report:")
print(class_report_adv)

# Evaluate fairness
sensitive_test_tensor = tf.convert_to_tensor(sensitive_test, dtype=tf.float32)
adversary_predictions_test = adversary(predictor(tf.convert_to_tensor(X_test_adv_scaled, dtype=tf.float32)))

dp_diff_adv = demographic_parity_difference(y_test_adv, y_pred_adv, sensitive_features=sensitive_test)
eo_diff_adv = equalized_odds_difference(y_test_adv, y_pred_adv, sensitive_features=sensitive_test)

print("Fairness Metrics for Adversarial Debiasing:")
print(f"Demographic Parity Difference: {dp_diff_adv:.4f}")
print(f"Equalized Odds Difference: {eo_diff_adv:.4f}")

# Calculate regular metrics for Adversarial Debiasing
y_proba_adv = predictor.predict(X_test_adv_scaled).flatten()
adv_metrics = {
    'Accuracy': accuracy_score(y_test_adv, y_pred_adv),
    'Precision': precision_score(y_test_adv, y_pred_adv),
    'Recall': recall_score(y_test_adv, y_pred_adv),
    'F1-Score': f1_score(y_test_adv, y_pred_adv),
    'ROC AUC': roc_auc_score(y_test_adv, y_proba_adv)
}

# PART 11 - Comparing All Methods

# Comparison table...
fairness_results = {
    "Model": ["Baseline Logistic Regression", "SMOTE Logistic Regression", "Random Forest", "Adversarial Debiasing"],
    "Demographic Parity Difference": [dp_diff_baseline, dp_diff_smote, dp_diff_rf, dp_diff_adv],
    "Equalized Odds Difference": [eo_diff_baseline, eo_diff_smote, eo_diff_rf, eo_diff_adv]
}

fairness_df = pd.DataFrame(fairness_results)
print(fairness_df)

# Plotting fairness metrics
fairness_df.plot(x="Model", kind="bar", figsize=(10, 6), title="Fairness Metrics Comparison")
plt.ylabel("Metric Value")
plt.xticks(rotation=0)
plt.legend(loc="upper left")
plt.show()


# METRICS
# Combine all metrics into a DataFrame for easier comparison
import pandas as pd

metrics_df = pd.DataFrame({
    'Baseline Logistic Regression': baseline_metrics,
    'SMOTE-Enhanced Logistic Regression': smote_metrics,
    'Random Forest with SMOTEENN': rf_metrics,
    'Adversarial Debiasing': adv_metrics
})
metrics_df = metrics_df.T
print(metrics_df)

# Plot metrics comparison for each model
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0, fontsize=7)
plt.show()

# Reset index for predictions to align with test set indices
y_pred_rf_series = pd.Series(y_pred_rf, index=X_test.index)

# Identify misclassified cases for the Random Forest model
misclassified_indices = X_test.index[(y_test != y_pred_rf_series)]
misclassified_samples = X_test.loc[misclassified_indices]
misclassified_true_labels = y_test.loc[misclassified_indices]
misclassified_pred_labels = y_pred_rf_series.loc[misclassified_indices]

# Confusion Matrix Heatmap for Baseline Model
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Baseline Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

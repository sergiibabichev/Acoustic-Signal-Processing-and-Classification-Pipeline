from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from signals_classification_functions import (load_signals, standardize_signals,
                                              segment_signals, extract_features_from_segments,
                                              rf_cv)

# Data loading
filepaths = [
    'filtered_signal_1.csv', 'filtered_signal_2.csv', 'filtered_signal_3.csv',
    'filtered_signal_4.csv', 'filtered_signal_5.csv',
]
labels = [0, 1, 2, 3, 4]  # Define labels for each file
signals = load_signals(filepaths, labels)

# Signals scaling
standardized_signals = standardize_signals(signals)

# Segment signals into chunks of 1000 points
data = segment_signals(signals, segment_size=1000)

# Features extraction
segmented_data = [(row[:-1], row[-1]) for row in data]
features_df = extract_features_from_segments(segmented_data)

# Transform labels into numeric
X = features_df.drop(columns=['label'])
y = features_df['label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random forest parameters optimization
param_bounds = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
}
optimizer = BayesianOptimization(
    f=lambda **params: rf_cv(X_train, y_train, **params),
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=10, n_iter=30)

# Train the final Random Forest classifier
best_params = optimizer.max['params']
best_params = {
    'n_estimators': int(best_params['n_estimators']),
    'max_depth': None if best_params['max_depth'] < 1 else int(best_params['max_depth']),
    'min_samples_split': int(best_params['min_samples_split']),
    'min_samples_leaf': int(best_params['min_samples_leaf']),
}
clf = RandomForestClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
feature_importances.head(11).plot(kind='bar')
plt.title("Top 11 Features")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.savefig("feature_importance.png", dpi=600, bbox_inches='tight')
plt.show()

# Train and Evaluate with Top 11 Features
top_11_features = feature_importances.head(11).index
X_train_top11 = X_train[top_11_features]
X_test_top11 = X_test[top_11_features]

clf_top11 = RandomForestClassifier(**best_params, random_state=42)
clf_top11.fit(X_train_top11, y_train)
y_pred_top11 = clf_top11.predict(X_test_top11)

print("Classification Report (Top 11 Features):\n", classification_report(y_test, y_pred_top11))

# Visualize the confusion matrix for the top 11 features
cm_top11 = confusion_matrix(y_test, y_pred_top11)
disp_top11 = ConfusionMatrixDisplay(confusion_matrix=cm_top11, display_labels=label_encoder.classes_)
disp_top11.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Top 11 Features)")
plt.show()


# Create a single figure with two confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # One row, two columns

# Plot confusion matrix for full feature set
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title("Confusion Matrix (All Features)")

# Plot confusion matrix for top 11 features
disp_top11 = ConfusionMatrixDisplay(confusion_matrix=cm_top11, display_labels=label_encoder.classes_)
disp_top11.plot(ax=axes[1], cmap=plt.cm.Blues)
axes[1].set_title("Confusion Matrix (Top 11 Features)")

# Adjust layout and save as PNG with 600 DPI
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=600, bbox_inches='tight')
plt.show()



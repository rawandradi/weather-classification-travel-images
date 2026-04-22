import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


X = np.load("X_weather_real.npy")
y = np.load("y_weather_real.npy")

#split data
# split test set (20%)
X_train, X_temp, weather_train, weather_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

# split remaining into train and validation sets
# (25% of 80% = 20%) for validation set
# remaining 60% for training set 
X_val, X_test, weather_val, weather_test = train_test_split(
    X_temp, weather_temp, test_size=0.50, random_state=42, stratify=weather_temp
)

# normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# HYPERPARAMETER TUNING FOR C controls the margin
# logarithmic grid search 0.01 x10 -> 0.1 x10 -> 1 x10 -> 10
C_values = [0.01, 0.1, 1, 10]
val_results = {}

for C in C_values:
    svm = SVC(C=C, kernel='rbf', gamma='scale') # RBF kernel for Non-linear patterns
    svm.fit(X_train, weather_train)

    weather_val_pred = svm.predict(X_val)
    acc = accuracy_score(weather_val, weather_val_pred)

    val_results[C] = acc
    print(f"C = {C}  → Validation Accuracy = {acc:.4f}")

# Select best C
best_C = max(val_results, key=val_results.get)
print("Best C based on validation:", best_C)

# Train final model with best C
X_final_train = np.vstack((X_train, X_val))
weather_final_train = np.hstack((weather_train, weather_val))

# class_weight='balanced' : Rare class → larger weight, Frequent class → smaller weight
svm_final = SVC(C=best_C, kernel='rbf', gamma='scale', class_weight='balanced')
svm_final.fit(X_final_train, weather_final_train)

# Evaluate on test set
weather_test_pred = svm_final.predict(X_test)

print("Final weather test Accuracy:", accuracy_score(weather_test, weather_test_pred)) #weather_test: actual labels, weather_test_pred: predicted labels
print("\nWeather Classification Report:\n")
# weather_test: actual labels, weather_test_pred: predicted labels, target_names: list of weather categories
#precision, recall, F1-score, and support for each class 
print(classification_report(weather_test, weather_test_pred, zero_division=0))


# Confusion Matrix

cm = confusion_matrix(weather_test, weather_test_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Weather Prediction Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("svm_weather_confusion_matrix.png")
plt.show()







import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('cardio_train.csv', delimiter=';')

def days_to_years(row):
    return int(row / 365)

df['age'] = df["age"].apply(days_to_years)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:-1], df['cardio'], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a **Logistic Regression** model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# **Step 2: Compute Evaluation Metrics**
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# **Step 3: Display Metrics**
print("\n✅ Model Evaluation Results:")
print(f"📌 Accuracy: {accuracy:.4f}")
print(f"📌 Precision: {precision:.4f}")
print(f"📌 Recall: {recall:.4f}")
print(f"📌 F1 Score: {f1:.4f}")

# **Step 4: Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred)

# **Step 5: Plot Confusion Matrix**
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# # Save model and scaler
# with open("model.pkl", "wb") as model_file:
#     pickle.dump(model, model_file)

# with open("scaler.pkl", "wb") as scaler_file:
#     pickle.dump(scaler, scaler_file)

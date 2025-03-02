#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load dataset
df = pd.read_csv("CloudWatch_Traffic_Web_Attack.csv")

# View basic info
print(df.info())
df.head()


# In[2]:


# Check for missing values
print(df.isnull().sum())

# Convert time columns to datetime
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize text data
df['src_ip_country_code'] = df['src_ip_country_code'].str.upper()

# Display updated DataFrame
df.info()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of bytes in and bytes out
plt.figure(figsize=(12, 6))
sns.histplot(df['bytes_in'], bins=50, color='blue', kde=True, label='Bytes In')
sns.histplot(df['bytes_out'], bins=50, color='red', kde=True, label='Bytes Out')
plt.legend()
plt.title('Distribution of Bytes In and Bytes Out')
plt.show()

# Suspicious activities by country
plt.figure(figsize=(15, 8))
sns.countplot(y=df['src_ip_country_code'], order=df['src_ip_country_code'].value_counts().index)
plt.title('Interaction Count by Source IP Country Code')
plt.show()


# In[4]:


df['session_duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
df['avg_packet_size'] = (df['bytes_in'] + df['bytes_out']) / df['session_duration']
df.head()


# In[5]:


from sklearn.ensemble import IsolationForest

# Selecting features for anomaly detection
features = df[['bytes_in', 'bytes_out', 'session_duration', 'avg_packet_size']]

# Initialize the model
model = IsolationForest(contamination=0.05, random_state=42)

# Fit and predict anomalies
df['anomaly'] = model.fit_predict(features)
df['anomaly'] = df['anomaly'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')

# Check the proportion of anomalies detected
print(df['anomaly'].value_counts())


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Convert anomaly labels to binary (1 = Suspicious, 0 = Normal)
df['is_suspicious'] = df['anomaly'].apply(lambda x: 1 if x == 'Suspicious' else 0)

# Select features and labels
X = df[['bytes_in', 'bytes_out', 'session_duration', 'avg_packet_size']]
y = df['is_suspicious']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = rf_classifier.predict(X_test)

# Evaluate performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Neural Network model
model = Sequential([
    Dense(8, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=8, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = df[['bytes_in', 'bytes_out', 'session_duration', 'avg_packet_size']].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[9]:


df.set_index('creation_time', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['bytes_in'], label='Bytes In', marker='o', alpha=0.7)
plt.plot(df.index, df['bytes_out'], label='Bytes Out', marker='o', alpha=0.7)
plt.title('Web Traffic Analysis Over Time')
plt.xlabel('Time')
plt.ylabel('Bytes')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[10]:


import networkx as nx

# Create a network graph
G = nx.Graph()

# Add edges from source IP to destination IP
for _, row in df.iterrows():
    G.add_edge(row['src_ip'], row['dst_ip'])

# Draw the network graph
plt.figure(figsize=(14, 10))
nx.draw_networkx(G, with_labels=True, node_size=20, font_size=8, node_color='skyblue')
plt.title('Network Interaction between Source and Destination IPs')
plt.axis('off')
plt.show()


# In[11]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the best model
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Evaluate
y_pred = best_rf.predict(X_test)
print("Optimized Model Accuracy:", accuracy_score(y_test, y_pred))


# In[12]:


from tensorflow.keras.layers import Dropout, BatchNormalization

# Define Neural Network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Optimized Neural Network Accuracy: {accuracy*100:.2f}%")


# In[13]:


from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = best_rf.predict(features)
    result = "Suspicious" if prediction[0] == 1 else "Normal"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)


# In[14]:


import streamlit as st

st.title("Web Threat Detection Dashboard")

# User Inputs
bytes_in = st.number_input("Bytes In")
bytes_out = st.number_input("Bytes Out")
session_duration = st.number_input("Session Duration")
avg_packet_size = st.number_input("Avg Packet Size")

# Prediction
if st.button("Predict"):
    pred = best_rf.predict([[bytes_in, bytes_out, session_duration, avg_packet_size]])
    st.write("Result: **Suspicious**" if pred[0] == 1 else "Result: **Normal**")


# In[ ]:





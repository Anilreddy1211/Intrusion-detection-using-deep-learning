import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# Ensure directories exist
os.makedirs("scaler_encoder", exist_ok=True)

def load_data():
    """Loads training and testing datasets."""
    train = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\intrusion_detection\\intrusion_detection\\datasets\\Train_data_1.csv")
    test = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\intrusion_detection\\intrusion_detection\\datasets\\Test_data_1.csv")

    # Drop redundant column
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    return train, test

def preprocess_data(train, test):
    """Standardizes numerical features and encodes categorical features."""
    num_cols = train.select_dtypes(include=['float64', 'int64']).columns

    # Standardize numerical features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[num_cols])
    test_scaled = scaler.transform(test[num_cols])

    # Save the scaler
    joblib.dump(scaler, "scaler.pkl")

    # Convert back to DataFrame
    train_scaled_df = pd.DataFrame(train_scaled, columns=num_cols)
    test_scaled_df = pd.DataFrame(test_scaled, columns=num_cols)

    # Encode categorical features
    encoder = LabelEncoder()
    cat_train = train.select_dtypes(include=['object']).copy()
    cat_test = test.select_dtypes(include=['object']).copy()

    if 'class' in cat_train.columns:
        y_train = encoder.fit_transform(cat_train['class'])
        cat_train.drop(['class'], axis=1, inplace=True)
    else:
        raise ValueError("Error: 'class' column is missing from training data")

    for col in cat_train.columns:
        cat_train[col] = encoder.fit_transform(cat_train[col])
        cat_test[col] = cat_test[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    # Save encoder
    joblib.dump(encoder, "encoder.pkl")

    return train_scaled_df, cat_train, test_scaled_df, cat_test, y_train

# Load and preprocess data
train, test = load_data()
train_scaled_df, cat_train, test_scaled_df, cat_test, y_train = preprocess_data(train, test)

# Combine numerical and categorical features
train_x = pd.concat([train_scaled_df, cat_train], axis=1)
test_x = pd.concat([test_scaled_df, cat_test], axis=1)

# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes)

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(train_x, y_train_categorical, train_size=0.70, random_state=42)

# Save processed data
joblib.dump((X_train, X_val, Y_train, Y_val), "data.pkl")

print("✅ Preprocessing Complete. Data Ready for Model Training.")

# Function to build model
def build_model(model_type="LSTM"):
    """Builds an LSTM or GRU model for cyberattack detection."""
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=False))
    else:
        model.add(GRU(128, input_shape=(X_train.shape[1], 1), return_sequences=False))

    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    return model

# Train and save LSTM model
lstm_model = build_model("LSTM")
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=2, batch_size=32)
lstm_model.save("scaler_encoder/lstm_model.h5")

# Train and save GRU model
gru_model = build_model("GRU")
gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
gru_history = gru_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=2, batch_size=32)
gru_model.save("scaler_encoder/gru_model.h5")

print("✅ Model Training Completed.")

# Load trained models
lstm_model = tf.keras.models.load_model("scaler_encoder/lstm_model.h5")
gru_model = tf.keras.models.load_model("scaler_encoder/gru_model.h5")

# Function to plot loss curves
def plot_loss_curves():
    """Plots the training and validation loss curves for LSTM and GRU models."""
    # Get the actual number of epochs from the history
    epochs = range(1, len(lstm_history.history['loss']) + 1) 
    lstm_train_loss = lstm_history.history['loss']
    lstm_val_loss = lstm_history.history['val_loss']
    gru_train_loss = gru_history.history['loss']
    gru_val_loss = gru_history.history['val_loss']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LSTM Loss Plot
    axes[0].plot(epochs, lstm_train_loss, label='Train')
    axes[0].plot(epochs, lstm_val_loss, label='Validation')
    axes[0].set_title('LSTM Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # GRU Loss Plot
    axes[1].plot(epochs, gru_train_loss, label='Train')
    axes[1].plot(epochs, gru_val_loss, label='Validation')
    axes[1].set_title('GRU Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
plot_loss_curves()

# Function to evaluate model
def evaluate_model(model, model_name):
    """Evaluates a model and plots the confusion matrix."""
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(Y_val, axis=1)

    print(f"\n====== {model_name} Model Evaluation ======")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="coolwarm", fmt="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

evaluate_model(lstm_model, "LSTM")
evaluate_model(gru_model, "GRU")

# Plot class distribution (bar chart)
plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='class', order=train['class'].value_counts().index, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Attack Class Distribution")
plt.show()

# Plot class distribution (pie chart)
class_counts = train['class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=sns.color_palette("coolwarm", len(class_counts)))
plt.title("Proportion of Attack Types")
plt.show()

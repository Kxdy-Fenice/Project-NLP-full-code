import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D,
                                     GlobalMaxPooling1D, Bidirection, LSTM, Dense, Activation, Concatenate)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import pickle
import os

# Seed for reproducibility
seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.config.experimental.enable_op_determinism()

# --- 1. Data Loading and Preprocessing ---
# UCI SMS Spam Collection
# url: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
sms_data_path = 'Data/sms+spam+collection(1)/SMSSpamCollection'

try:
    df = pd.read_csv(sms_data_path, sep='\t', header=None, names=['label', 'text'], encoding='latin-1')

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    texts = df['text'].values
    labels = df['label'].values

except FileNotFoundError:
    print(f"Error: SMSSpamCollection file not found at {sms_data_path}. Please download it.")
    exit()
except Exception as e:
    print(f"An error occurred while loading/processing SMSSpamCollection: {e}")
    exit()

# Parameters for text preprocessing
vocab_size = 1000
max_sequence_length = 150

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(texts)

padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=labels)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Testing labels shape: {y_test.shape}")

# --- 2. Model definition ---
# Model hyperparameters
embedding_dim = 128
cnn_filters = 128
cnn_kernel_size = 5
lstm_units = 64

input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_sequence_length)(input_layer)

cnn_output = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu')(embedding_layer)
cnn_pooled_output = GlobalMaxPooling1D()(cnn_output)

bilstm_output = Bidirection(LSTM(units=lstm_units, return_sequences=False))(embedding_layer)

# concatenate CNN and BiLSTM branches
merged_output = Concatenate()([cnn_pooled_output, bilstm_output])

# first dense layer with tanh activation
dense_tanh = Dense(64)(merged_output)
activated_tanh = Activation('tanh')(dense_tanh)

output_layer = Dense(1)(activated_tanh)
activated_sigmoid = Activation('sigmoid')(output_layer)

# create model
model = Model(inputs=input_layer, outputs=activated_sigmoid)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary

# --- 3. Model Training ---
batch_size = 32
epochs = 50

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1)

# Save trained model and tokenizer
model_filename = 'UCI_cnn_bilstm_model.keras'
model.save(model_filename)
print(f"\nModel saved to {model_filename}")

tokenizer_filename = 'UCI_tokenizer.pkl'
with open(tokenizer_filename, 'wb') as file:
    pickle.dump(tokenizer, file)
print(f"Tokenizer saved to {tokenizer_filename}")

# --- 4. Model Evaluation ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n--- Classification Report (UCI SMS Test Data) ---")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("\n--- Confusion Matrix (UCI SMS Test Data) ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

roc_auc = roc_auc_score(y_test, y_pred_probs)
print(f"\nROC AUC Score: {roc_auc:.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', alpha=0.8, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (UCI SMS Test Data)')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

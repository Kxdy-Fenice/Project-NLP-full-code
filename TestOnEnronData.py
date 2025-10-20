import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import pickle
import re

# Load Enron Dataset
# url: https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data
enron_spam_csv_path = 'Data/enron_spam_data.csv'

vocab_size = 10000
max_sequence_length = 150

try:
    model_filename = 'Model/UCI_cnn_bilstm_model.keras'
    loaded_model = load_model(model_filename)
    print(f"Model loaded successfully from {model_filename}")

    tokenizer_filename = 'Model/UCI_tokenizer.pkl'
    with open(tokenizer_filename, 'rb') as file:
        loaded_tokenizer = pickle.load(file)
    print(f"Tokenizer loaded successfully from {tokenizer_filename}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Please ensure '{model_filename}' and '{tokenizer_filename}' are in the correct directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    print("This might indicate a TensorFlow/Keras version mismatch or a corrupted file. Check your versions.")
    import traceback
    traceback.print_exc()
    exit()


# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(
        r'(from|to|subject|'
        r'date|message-id|mime-version|content-type|'
        r'content-transfer-encoding|x-from|x-to|x-cc|'
        r'x-bcc|x-folder|x-origin|x-filename):.*\n',
        '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load and Preprocess Enron Spam Test Data
try:
    df_enron = pd.read_csv(enron_spam_csv_path, encoding='latin-1')

    if 'Subject' not in df_enron.columns or 'Message' not in df_enron.columns or 'Spam/Ham' not in df_enron.columns:
        raise KeyError("Expected columns 'Subject', 'Message', and 'Spam/Ham' not found in the CSV.")

    df_enron['full_text'] = df_enron['Subject'].fillna('') + ' ' + df_enron['Message'].fillna('')
    df_enron['label'] = df_enron['Spam/Ham'].map({'ham': 0, 'spam': 1})
    df_enron.dropna(subset=['label', 'full_text'], inplace=True)
    df_enron['label'] = df_enron['label'].astype(int)

    enron_texts = df_enron['full_text'].values
    enron_labels = df_enron['label'].values

except FileNotFoundError:
    print(f"Error: Enron Spam Data not found at {enron_spam_csv_path}. Please check the path and download it.")
    exit()
except KeyError as e:
    print(f"Error: {e}")
    print("Column name issue. Please ensure the exact column names are 'Subject', 'Message', and 'Spam/Ham'.")
    print(f"Available columns: {df_enron.columns.tolist()}")
    exit()
except Exception as e:
    print(f"An error occurred while loading/processing Enron CSV: {e}")
    import traceback
    traceback.print_exc()
    exit()

cleaned_enron_texts = np.array([clean_text(str(text)) for text in enron_texts])

# Tokenize Enron test data using the same loaded tokenizer
test_sequences_enron = loaded_tokenizer.texts_to_sequences(cleaned_enron_texts)

X_test_enron = pad_sequences(test_sequences_enron, maxlen=max_sequence_length, padding='post', truncating='post')
y_test_enron = enron_labels

print(f"Enron Test data shape (after preprocessing): {X_test_enron.shape}")
print(f"Enron Test labels shape: {y_test_enron.shape}")
print(f"Number of ham samples: {np.sum(y_test_enron == 0)}")
print(f"Number of spam samples: {np.sum(y_test_enron == 1)}")

# Evaluate Pre-trained Model on the Enron Test Data
batch_size_eval = 32
loss, accuracy = loaded_model.evaluate(X_test_enron, y_test_enron, verbose=0, batch_size=batch_size_eval)
print(f"Enron Test Loss: {loss:.4f}")
print(f"Enron Test Accuracy: {accuracy:.4f}")

y_pred_probs_enron = loaded_model.predict(X_test_enron, batch_size=batch_size_eval).flatten()
y_pred_enron = (y_pred_probs_enron > 0.5).astype(int)

print("\n--- Classification Report (Enron Test Data) ---")
print(classification_report(y_test_enron, y_pred_enron, target_names=['Ham', 'Spam']))

print("\n--- Confusion Matrix (Enron Test Data) ---")
cm_enron = confusion_matrix(y_test_enron, y_pred_enron)
print(cm_enron)

roc_auc_enron = roc_auc_score(y_test_enron, y_pred_probs_enron)
print(f"\nEnron ROC AUC Score: {roc_auc_enron:.4f}")

precision_enron, recall_enron, _ = precision_recall_curve(y_test_enron, y_pred_probs_enron)
pr_auc_enron = auc(recall_enron, precision_enron)

print(f"Enron Precision-Recall AUC: {pr_auc_enron:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall_enron, precision_enron, color='g', alpha=0.8, lw=2, label=f'PR curve (area = {pr_auc_enron:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Enron Spam Data)')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

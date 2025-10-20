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
import os
import os

# Kaggle Dataset by Satyajeet Bedi
# url: https://www.kaggle.com/datasets/satyajeetbedi/email-hamspam-dataset
kaggle_csv_path = 'Data/email_spam.csv'

vocab_size = 10000
max_sequence_length = 150

# Load pre-trained model and tokenizer
try:
    model_filename = 'UCI_cnn_bilstm_model.keras'
    loaded_model = load_model(model_filename)
    print(f"Model loaded successfully from {model_filename}")

    tokenizer_filename = 'UCI_tokenizer.pkl'
    with open(tokenizer_filename, 'rb') as file:
        loaded_tokenizer = pickle.load(file)
    print(f"Tokenizer loaded successfully from {tokenizer_filename}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()


# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load and Preprocess Kaggle Test Data
try:
    df_email = pd.read_csv(kaggle_csv_path, encoding='latin-1')

    # Rename columns
    df_email.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

    # Map 'ham' and 'spam' string labels to 0 and 1
    df_email['label'] = df_email['label'].map({'ham': 0, 'spam': 1})

    df_email = df_email.loc[:, ['label', 'text']].copy()

    email_texts = df_email['text'].values
    email_labels = df_email['label'].values

except FileNotFoundError:
    print(f"Error: Kaggle Email Ham.Spam Dataset not found at {kaggle_csv_path}. Please check the path and download it.")
    exit()
except KeyError as e:
    print(f"Error: {e}")
    print(f"Available columns: {df_email.columns.tolist()}")
    exit()
except Exception as e:
    print(f"An error occurred while loading/processing Kaggle CSV: {e}")
    exit()

cleaned_email_texts = np.array([clean_text(str(text)) for text in email_texts])

test_sequences_email = loaded_tokenizer.texts_to_sequences(cleaned_email_texts)

X_test_email = pad_sequences(test_sequences_email, maxlen=max_sequence_length, padding='post', truncating='post')
y_test_email = email_labels

print(f"Kaggle Email Test data shape (after preprocessing): {X_test_email.shape}")
print(f"Kaggle Email Test labels shape: {y_test_email.shape}")

batch_size_eval = 32
loss, accuracy = loaded_model.evaluate(X_test_email, y_test_email, verbose=0, batch_size=batch_size_eval)
print(f"Kaggle Email Test Loss: {loss:.4f}")
print(f"Kaggle Email Test Accuracy: {accuracy:.4f}")

y_pred_probs_email = loaded_model.predict(X_test_email, batch_size=batch_size_eval).flatten()
y_pred_email = (y_pred_probs_email > 0.5).astype(int)

print("\n--- Classification Report (Kaggle Email Test Data) ---")
print(classification_report(y_test_email, y_pred_email, target_names=['Ham', 'Spam']))

print("\n--- Confusion Matrix (Kaggle Email Test Data) ---")
cm_email = confusion_matrix(y_test_email, y_pred_email)
print(cm_email)

roc_auc_email = roc_auc_score(y_test_email, y_pred_probs_email)
print(f"\nKaggle Email ROC AUC Score: {roc_auc_email:.4f}")

precision_email, recall_email, _ = precision_recall_curve(y_test_email, y_pred_probs_email)
pr_auc_email = auc(recall_email, precision_email)

print(f"Kaggle Email Precision-Recall AUC: {pr_auc_email:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall_email, precision_email, color='b', alpha=0.8, lw=2, label=f'PR curve (area = {pr_auc_email:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Satyajeet Bedi Email Ham.Spam Test Data)')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

max_sequence_length = 150
seed_value = 42

model_path = 'Model/UCI_cnn_bilstm_model.keras'
tokenizer_path = 'Model/UCI_tokenizer.pkl'


# Text Cleaning Function
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


# Load Pre-trained Model and Tokenizer
try:
    loaded_model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

    with open(tokenizer_path, 'rb') as file:
        loaded_tokenizer = pickle.load(file)
    print(f"Tokenizer loaded successfully from {tokenizer_path}")

except FileNotFoundError:
    print(f"Error: Model or tokenizer file not found.")
    print(f"Please ensure '{model_path}' and '{tokenizer_path}' are in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    print("This might indicate a TensorFlow/Keras version mismatch, or issues with the file itself.")
    exit()


# Function to predict whether an input is 'ham' or 'spam'
def predict_spam_ham(input_text, model, tokenizer, max_sequence_length, threshold=0.5):
    """
    Predicts whether a given input text is ham or spam using the loaded model.

    Args:
        input_text (str): The raw text string to classify
        model (tf.keras.Model): The loaded Keras model
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The loaded tokenizer
        max_sequence_length (int): The maximum sequence length used during training
        threshold (float): Probability threshold for classifying as spam (default 0.5)

    Returns:
        tuple: A tuple containing (classification_label, spam_probability).
    """
    cleaned_text = clean_text(input_text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])  # Input needs to be a list

    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')

    prediction_probability = model.predict(padded_sequence)[0][0]

    if prediction_probability >= threshold:
        classification = "SPAM"
    else:
        classification = "HAM"

    return classification, prediction_probability


# Example Usage
# Spam Example
spam_message = "WINNER! You have won a FREE iPhone! Click now to claim your prize: http://freeiphone.com/phone-prize"
label, prob = predict_spam_ham(spam_message, loaded_model, loaded_tokenizer, max_sequence_length)
print(f"Input: '{spam_message}'")
print(f"Prediction: {label} (Probability of Spam: {prob:.4f})\n")

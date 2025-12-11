import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model and tokenizer
model = load_model("next_word_lstm.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, input_text, max_sequence_length):
    input_text = input_text.lower()
    token_list = tokenizer.texts_to_sequences([input_text])[0]

    if not token_list:
        return "No valid tokens found."

    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Reverse lookup the predicted index
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return "Word not found in tokenizer."

# Streamlit UI
st.title("Next Word Prediction with LSTM RNN")

input_text = st.text_input("Enter your text:", "To be or not to be")

if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write(f"### Predicted Next Word: **{next_word}**")

import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load Model
# -------------------------------
model = load_model("next_word_lstm.keras")

# -------------------------------
# Load Tokenizer
# -------------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------------
# Load Configuration
# -------------------------------
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]

# -------------------------------
# Title
# -------------------------------
st.title("🤖 Next Word Prediction using LSTM")

st.write("Enter a sentence and click Predict.")

# -------------------------------
# User Input
# -------------------------------
text = st.text_input("Enter a sentence")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Next Word"):

    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:

        # Convert sentence into numbers
        token_text = tokenizer.texts_to_sequences([text])[0]

        # Padding
        padded = pad_sequences(
            [token_text],
            maxlen=max_len-1,
            padding="pre"
        )

        # Predict
        prediction = model.predict(padded, verbose=0)

        predicted_index = np.argmax(prediction)

        predicted_word = ""

        # Convert index back to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        st.success("Predicted Sentence")

        st.markdown(f"## {text} **{predicted_word}**")
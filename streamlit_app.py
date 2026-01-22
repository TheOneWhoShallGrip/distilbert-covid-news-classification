import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import os

st.set_page_config(
    page_title="Covid News Detector with DistilBERT",
    layout="centered",
)

#-----------------------------------------------------------------------------------------------------

# Rebuild the model architecture
model = TFAutoModel.from_pretrained("distilbert/distilbert-base-uncased", from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        return self.fc(x)

classifier = BERTForClassification(model)

# needs to have this or it breaks idk why
dummy_inputs = {
    "input_ids": tf.zeros((1, 100), dtype=tf.int32),
    "attention_mask": tf.zeros((1, 100), dtype=tf.int32)
}
classifier(dummy_inputs)

# Check if weight file exists or not
weights_path = "model_weights.h5"
if os.path.exists(weights_path):
    classifier.load_weights(weights_path)
else:
    st.warning("Warning: Weights file not found. Model will produce random outputs instead.")

#---------------------------------------------------------------------------------------------------

# UI
st.markdown("""
    <style>
    .stApp {
        background-color: #2e2e2e;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    textarea {
        background-color: #2c2c2c !important;
        color: #f5f5f5 !important;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        border: none;
        resize: vertical;
    }
    .stButton>button {
        background-color: #04AA6D;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-size: 16px;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #028a57;
        color: #f5f5f5;
        cursor: pointer;
    }
    .stTextArea>label {
        font-size: 18px;
        color: #f5f5f5;
        margin-bottom: 8px;
    }
    hr {
        border: 0.5px solid #444444;
    }
    </style>
""", unsafe_allow_html=True)

st.title("COVID-19 Fake News Detector")

st.markdown("""
    <p>This app detects whether a COVID-19 related news snippet is <strong>real or fake</strong> using a fine-tuned <strong>DistilBERT base</strong> model.</p>
    
    <p><strong>Input:</strong> Text snippet (up to 512 tokens)</p>
    <hr>
    """, unsafe_allow_html=True)

user_input = st.text_area(
    "Enter a news snippet below to classify its authenticity:",
    height=150,
    placeholder="Example: Vaccines contain microchips to track people..."
)

MAX_LENGTH = 512
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="tf",
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )

        prediction = classifier.predict({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })

        # confidence = prediction.item()
        # label = "Real" if confidence > 0.5 else "Fake"
        # st.success(f"Prediction: **{label}** ({confidence:.2%} confidence) | raw confidence: {confidence}")

        confidence = prediction.item()
        label = "Real" if confidence > 0.5 else "Fake"
        bg_color = "#569143" if confidence > 0.5 else "#ba4c6e"

        st.markdown(f"""
        <div style='
            background-color: {bg_color};
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            margin-top: 20px;
        '>
            Prediction: <strong>{label}!</strong><br>
            Confidence (Real): {confidence:.2%}
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <hr style="margin-top: 50px;">
    <p style='text-align: center; color: grey;'>
        Built using Streamlit | ⚠️ Please verify important info from trusted sources! | <a href="https://github.com/Shardium/COVID-News-Classifier-with-BERT" style="color: #4ecdc4; text-decoration: none;" target="_blank">GitHub</a>
    </p>

""", unsafe_allow_html=True)

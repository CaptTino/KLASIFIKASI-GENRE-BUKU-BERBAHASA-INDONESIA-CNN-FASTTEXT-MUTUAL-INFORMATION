import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# =====================
# SETUP
# =====================
st.set_page_config(
    page_title="Klasifikasi Genre Buku",
    page_icon="📚",
    layout="centered"
)

nltk.download('stopwords')

# =====================
# LOAD ASSETS
# =====================
@st.cache_resource
def load_assets():
    model = load_model("cnn_fasttext_mi.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("mi_features.pkl", "rb") as f:
        mi_features = set(pickle.load(f))

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, label_encoder, mi_features, config


model, tokenizer, label_encoder, mi_features, config = load_assets()
max_len = config["max_len"]

# =====================
# PREPROCESSING
# =====================
stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def apply_mi_filter(tokens, mi_features):
    filtered = [t for t in tokens if t in mi_features]
    return " ".join(filtered)

# =====================
# UI
# =====================
st.title("📚 Klasifikasi Genre Buku")
st.write(
    "Aplikasi ini menggunakan **CNN + FastText + Mutual Information** "
    "untuk memprediksi genre buku berdasarkan sinopsis."
)

user_input = st.text_area(
    "✍️ Masukkan Sinopsis Buku",
    height=220,
    placeholder="Contoh: Kisah petualangan seorang pemuda di kerajaan kuno yang penuh sihir..."
)

if st.button("🔍 Prediksi Genre"):
    if user_input.strip() == "":
        st.warning("Sinopsis tidak boleh kosong.")
    else:
        # 1. Preprocessing
        tokens = preprocess_text(user_input)

        # 2. Mutual Information Filtering
        filtered_text = apply_mi_filter(tokens, mi_features)

        if filtered_text.strip() == "":
            st.error(
                "Kata-kata pada sinopsis tidak termasuk fitur penting hasil Mutual Information."
            )
        else:
            # 3. Tokenizing & Padding
            seq = tokenizer.texts_to_sequences([filtered_text])
            pad = pad_sequences(seq, maxlen=max_len, padding="post")

            # 4. Prediction
            prediction = model.predict(pad)
            pred_idx = np.argmax(prediction)
            genre = label_encoder.inverse_transform([pred_idx])[0]
            confidence = np.max(prediction) * 100

            # 5. Output
            st.success(f"🎯 Genre Buku: **{genre}**")
            st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

            with st.expander("🔎 Detail Proses"):
                st.write("**Hasil Preprocessing:**")
                st.code(" ".join(tokens))
                st.write("**Setelah Seleksi Mutual Information:**")
                st.code(filtered_text)

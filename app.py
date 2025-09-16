import streamlit as st
import joblib
import gzip
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

st.set_page_config(page_title="Comment Sentiment Analysis", page_icon="💬")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inject custom CSS for modern, creative look
st.markdown('''
    <style>
    body {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    .stApp {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    /* Ẩn header Streamlit */
    header[data-testid="stHeader"], .st-emotion-cache-18ni7ap.ezrtsby0 {
        display: none !important;
    }
    /* Ẩn menu hamburger Streamlit */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #234567;
        text-align: center;
        margin-bottom: 0.5em;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px #c2e9fb;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #234567;
        text-align: center;
        margin-bottom: 2em;
    }
    .result-box {
        background: rgba(255,255,255,0.85);
        border-radius: 22px;
        padding: 1.2em 1.2em 0.8em 1.2em;
        margin: 1em auto;
        box-shadow: 0 4px 16px 0 #5b86e544;
        border: 2px solid #a1c4fd;
        max-width: 260px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .emoji {
        font-size: 2.8rem;
        margin-bottom: 0.2em;
        filter: drop-shadow(0 2px 8px #a1c4fd88);
    }
    .result-sentiment {
        font-size: 1.3em;
        font-weight: bold;
        color: #5b86e5;
        margin-top: 0.2em;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #a1c4fd33;
    }
    .stTextArea textarea {
        background: #fffbe6;
        border-radius: 10px;
        font-size: 1.1rem;
        color: #234567;
        border: 2px solid #e0e0e0;
        transition: border 0.2s, box-shadow 0.2s;
    }
    .stTextArea textarea:focus {
        background: #fff;
        border: 2.5px solid #ff9966;
        box-shadow: 0 0 8px #ff996655;
        outline: none;
    }
    /* Custom multiselect */
    .stMultiSelect>div>div {
        background: #fff;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px #fda08533;
        padding: 6px 10px;
        font-size: 1.1rem;
        color: #234567;
        border: 1.5px solid #f6d365;
    }
    /* Custom tag for selected option */
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(90deg, #ffe082 0%, #ffd54f 100%);
        color: #234567 !important;
        border-radius: 8px !important;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 2px 4px;
        box-shadow: 0 1px 4px #fda08522;
        padding: 4px 12px;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: #234567 !important;
    }
    .stMultiSelect [aria-selected="true"] {
        background: #fffde7 !important;
    }
    .stMultiSelect input {
        font-size: 1.1rem;
        color: #234567;
    }
    .stButton>button {
        background: linear-gradient(90deg, #89f7fe 0%, #66a6ff 100%);
        color: #fff;
        font-weight: bold;
        border-radius: 10px;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px #66a6ff44;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%);
        color: #fff;
        transform: scale(1.04);
    }
    .result-title {
        color: #234567;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    label, .stTextInput label, .stSelectbox label, .stTextArea label {
        color: #234567 !important;
        font-weight: 600;
    }
    .main-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        width: 100%;
    }
    .form-box {
        background: rgba(255,255,255,0.92);
        border-radius: 28px;
        box-shadow: 0 8px 32px 0 #5b86e544;
        padding: 2.5em 2.5em 2em 2.5em;
        max-width: 480px;
        width: 100%;
        margin: 0 auto 2em auto;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .form-box label, .form-box .stTextArea label, .form-box .stSelectbox label {
        color: #234567 !important;
        font-weight: 600;
        font-size: 1.1em;
    }
    .form-box .stTextArea, .form-box .stSelectbox, .form-box .stButton {
        width: 100% !important;
        margin-bottom: 1.2em;
    }
    .form-box .stButton {
        display: flex;
        justify-content: center;
    }
    .result-title {
        margin-top: 0.5em;
    }
    @media (max-width: 600px) {
        .form-box {
            padding: 1.2em 0.5em 1em 0.5em;
            max-width: 98vw;
        }
        .result-box {
            max-width: 98vw;
        }
    }
    </style>
''', unsafe_allow_html=True)

# Load data và vectorizer
@st.cache_resource
def load_resources():
    train = pd.read_csv('data/processed_train.csv')
    tfidf_vec = TfidfVectorizer(max_features=10000).fit(train['text'])
    bog_vec = CountVectorizer(max_features=10000).fit(train['text'])
    binary_vec = CountVectorizer(max_features=10000, binary=True).fit(train['text'])
    # Load models
    k_means_model = joblib.load('weights/k_means.pkl')
    random_forest_model = joblib.load('weights/random_forest.pkl')
    logistic_model = joblib.load('weights/logistic_regression.pkl')
    with gzip.open("weights/knn.pkl.gz", "rb") as f:
        knn_model = joblib.load(f)
    with gzip.open("weights/svm.pkl.gz", "rb") as f:
        svm_model = joblib.load(f)
    # Load neural network (MLP) model
    import torch
    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=100, output_dim=3):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    input_dim = 10000
    nn_model = SimpleMLP(input_dim, hidden_dim=100)
    nn_model.load_state_dict(torch.load('weights/binary_mlp.pt', map_location='cpu'))
    nn_model.eval()
    # LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=256, input_length=24))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.load_weights('weights/lstm.h5')
    lstm_model = model
    total_word = 10000
    token = Tokenizer(num_words=total_word)
    token.fit_on_texts(train['text'])
    return {
        'tfidf_vec': tfidf_vec,
        'bog_vec': bog_vec,
        'binary_vec': binary_vec,
        'k_means_model': k_means_model,
        'random_forest_model': random_forest_model,
        'logistic_model': logistic_model,
        'knn_model': knn_model,
        'svm_model': svm_model,
        'lstm_model': lstm_model,
        'nn_model': nn_model,
        'token': token
    }
def transform_label(num):
    if num == 0:
        return 'negative'
    elif num == 1:
        return 'neutral'
    elif num == 2:
        return 'positive'
    else:
        raise ValueError(f"Invalid label: {num}. Chỉ chấp nhận 0, 1, 2.")
def inference(model_key, text, resources):
    text = word_process(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    tfidf_vec = resources['tfidf_vec']
    bog_vec = resources['bog_vec']
    binary_vec = resources['binary_vec']
    token = resources['token']
    if model_key == 'knn':
        x_predict = tfidf_vec.transform([text]).toarray()
        sentiment = transform_label(resources['knn_model'].predict(x_predict)[0])
    elif model_key == 'random_forest':
        x_predict = bog_vec.transform([text]).toarray()
        sentiment = transform_label(resources['random_forest_model'].predict(x_predict)[0])
    elif model_key == 'logistic_regression':
        x_predict = bog_vec.transform([text]).toarray()
        sentiment = transform_label(resources['logistic_model'].predict(x_predict)[0])
    elif model_key == 'svm':
        x_predict = tfidf_vec.transform([text]).toarray()
        sentiment = transform_label(resources['svm_model'].predict(x_predict)[0])
    elif model_key == 'kmeans':
        x_predict = tfidf_vec.transform([text]).toarray()
        sentiment = transform_label(resources['k_means_model'].predict(x_predict)[0])
    elif model_key == 'lstm':
        new_reviews_seq = token.texts_to_sequences([text])
        new_reviews_padded = pad_sequences(new_reviews_seq, maxlen=24)
        predictions = resources['lstm_model'].predict(new_reviews_padded, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        sentiment = transform_label(predicted_classes[0])
    elif model_key == 'nn':
        import torch
        x_predict = binary_vec.transform([text]).toarray()
        x_tensor = torch.tensor(x_predict, dtype=torch.float32)
        with torch.no_grad():
            output = resources['nn_model'](x_tensor)
            pred = torch.argmax(output, dim=1).item()
        sentiment = transform_label(pred)
    else:
        raise ValueError(f"Model {model_key} không hợp lệ.")
    return sentiment
def contains_vietnamese(text):
    # Regex kiểm tra ký tự tiếng Việt có dấu
    return bool(re.search(r'[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]', text, re.IGNORECASE))
def word_process(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_text)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_text)
# Header
st.markdown('<div class="main-header">🎉 Comment Sentiment Analysis 🎉</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">💡 Nhập một câu comment và chọn mô hình để phân tích cảm xúc. Hãy khám phá cảm xúc của bạn với giao diện trẻ trung và sáng tạo!</div>', unsafe_allow_html=True)

resources = load_resources()

# Label và input cho comment
st.markdown('''
    <div style="margin-bottom:0.15em;">
        <label style="font-weight:600;color:#234567;font-size:1.1em;display:inline-flex;align-items:center;">
            ✍️ Nhập comment:
        </label>
    </div>
''', unsafe_allow_html=True)
comment = st.text_area('', height=100)

model_options = {
    'KMeans': 'kmeans',
    'Random Forest': 'random_forest',
    'Logistic Regression': 'logistic_regression',
    'K-Nearest Neighbors': 'knn',
    'SVM': 'svm',
    'LSTM': 'lstm',
    'Neural Network': 'nn'
}

# Label và selectbox cho mô hình
st.markdown('''
    <div style="margin-bottom:0.15em;margin-top:0.7em;">
        <label style="font-weight:600;color:#234567;font-size:1.15em;display:inline-flex;align-items:center;">
            🤖 Chọn mô hình:
        </label>
    </div>
''', unsafe_allow_html=True)
selected_model = st.selectbox('', list(model_options.keys()), key='model_select')

# Icon cho từng cảm xúc
sentiment_icons = {
    'positive': '😄',
    'neutral': '😐',
    'negative': '😢'
}
sentiment_colors = {
    'positive': '#ffb347',
    'neutral': '#b0b0b0',
    'negative': '#ff6961'
}

if st.button('Phân tích cảm xúc'):
    if not comment.strip():
        st.markdown('''
            <div style="
                background: #fffbe6;
                border: 2px solid #ffe082;
                color: #b26a00;
                font-weight: 600;
                font-size: 1.15em;
                border-radius: 12px;
                padding: 1em 1.5em;
                margin: 1em auto;
                max-width: 420px;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;">
                <span style="font-size:1.5em;margin-right:0.5em;">⚠️</span> Vui lòng nhập comment!
            </div>
        ''', unsafe_allow_html=True)
    elif contains_vietnamese(comment):
        st.markdown('''
            <div style="
                background: #fffbe6;
                border: 2px solid #ffe082;
                color: #b26a00;
                font-weight: 600;
                font-size: 1.15em;
                border-radius: 12px;
                padding: 1em 1.5em;
                margin: 1em auto;
                max-width: 420px;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;">
                <span style="font-size:1.5em;margin-right:0.5em;">⚠️</span> Vui lòng chỉ nhập tiếng Anh, không nhập tiếng Việt có dấu!
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-title">Kết quả dự đoán:</div>', unsafe_allow_html=True)
        model_key = model_options[selected_model]
        sentiment = inference(model_key, comment, resources)
        if sentiment in sentiment_icons:
            icon = sentiment_icons[sentiment]
            color = sentiment_colors[sentiment]
            st.markdown(f'''
                <div class="result-box" style="border: 2.5px solid {color};">
                    <div class="emoji">{icon}</div>
                    <div class="result-sentiment" style="color: {color};">{sentiment.capitalize()}</div>
                </div>
            ''', unsafe_allow_html=True) 
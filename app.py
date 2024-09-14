from flask import Flask, render_template, request
import json
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
from datetime import datetime
import numpy as np

app = Flask(__name__)

#ambil data set
def load_data_from_json(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filename} tidak ditemukan.")
        return []

#simpan data
def save_data_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")

#F ambil data url & cleaning
def fetch_and_clean_text(url):
    try:
        response = requests.get(url)
        html_content = response.text

        #ambil data html
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()

        #cleansing(updated)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\W', ' ', text)
        text = text.lower()

        # Tokenisasi teks
        tokens = word_tokenize(text)

        # stopword Sastrawi
        factory = StopWordRemoverFactory()
        stop_words = set(factory.get_stop_words())
        tokens = [word for word in tokens if word not in stop_words]

        # Gabungkan tokens kembali menjadi teks
        cleaned_text = ' '.join(tokens)

        return cleaned_text, tokens

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return "", []
    except Exception as ex:
        print(f"Error: {ex}")
        return "", []

# jika tidak ada latih model
def create_and_train_model():
    data = load_data_from_json('urls.json')
    if not data:
        print("No training data found.")
        return

    X = [fetch_and_clean_text(item['url'])[0] for item in data]
    y = [item['label'] for item in data]

    vectorizer = TfidfVectorizer(max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X)

    model = SVC(kernel='linear', probability=False)
    model.fit(X_tfidf, y)

    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Muat model jika sudah ada
try:
    model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    create_and_train_model()
    model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    result_class = "hidden"
    word_count = 0
    char_count = 0
    frequent_words = {}
    explanation = ""

    if request.method == 'POST':
        user_url = request.form['url']
        if not user_url.startswith(('http://', 'https://')):
            user_url = 'https://' + user_url
        
        cleaned_text, tokens = fetch_and_clean_text(user_url)
        word_count = len(tokens)
        char_count = len(' '.join(tokens))

        # Hitung frekuensi kata
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        frequent_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:10])

        if cleaned_text:
            user_tfidf = vectorizer.transform([cleaned_text])
            user_prediction = model.predict(user_tfidf)

            if user_prediction[0] == 1:
                result = "Website Judi."
                result_class = "red"
                explanation = (f"Berdasarkan hasil ekstrak konten yang kami lakukan, sistem kami mendeteksi bahwa website "
                               f" yang anda masukan adalah website judi.")
            else:
                result = "Bukan Website Judi."
                result_class = "green"
                explanation = (f"Berdasarkan hasil ekstrak konten yang kami lakukan, sistem kami mendeteksi bahwa website "
                               f"yang anda masukan bukan website judi.")

            # Penjelasan
            feature_names = vectorizer.get_feature_names_out()
            feature_probs = np.zeros(len(feature_names))  # Kosongkan nilai jika predict_proba tidak ada
            sorted_features = sorted(zip(feature_names, feature_probs), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:10]

            explanation += " "

            # Simpan hasil prediksi ke file JSON
            data = load_data_from_json('urls.json')
            new_data = {
                "url": user_url,
                "label": int(user_prediction[0])
            }

            # Periksa apakah URL sudah ada di data
            if not any(d['url'] == user_url for d in data):
                data.append(new_data)
                save_data_to_json(data, 'urls.json')
        else:
            result = "Gagal mengambil teks dari URL."
            result_class = "yellow"
            explanation = "Gagal mengambil konten dari URL yang diberikan."

        # Simpan log hasil prediksi ke file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "url": user_url,
            "result": result,
            "word_count": word_count,
            "char_count": char_count,
            "frequent_words": frequent_words,
        }
        with open('logs.json', 'a') as log_file:
            json.dump(log_entry, log_file)
            log_file.write('\n')

    return render_template('index.html', result=result, result_class=result_class,
                           word_count=word_count, char_count=char_count,
                           frequent_words=frequent_words, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)

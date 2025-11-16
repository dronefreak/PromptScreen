import json
import pandas as pd
import string
import emoji
import nltk
from pathlib import Path
import joblib

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

_ = nltk.download('wordnet')
_ = nltk.download('stopwords')
_ = nltk.download('averaged_perceptron_tagger_eng')
_ = nltk.download('punkt')
_ = nltk.download('punkt_tab')

class TextPreProcessor:
    def __init__(self, custom_stopwords=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

    def _get_wordnet_pos(self, word: str):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = emoji.demojize(text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = word_tokenize(text)
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                pos = self._get_wordnet_pos(token)
                lemma = self.lemmatizer.lemmatize(token, pos)
                filtered_tokens.append(lemma)

        return ' '.join(filtered_tokens)

    def preprocess_df(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        df_copy = df.copy()
        output_column = 'clean_prompt'
        df_copy[output_column] = df_copy[text_column].apply(self.preprocess)
        return df_copy

class JailbreakClassifier:
    def __init__(self, json_file_path: str, model_output_dir: str = None):
        self.json_file_path = json_file_path
        self.model_output_dir = Path(model_output_dir) if model_output_dir else None
        self.preprocessor = TextPreProcessor()
        self.vectorizer = None
        self.model = None
        self.df = None

    def _load_and_preprocess_data(self):
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.df = pd.DataFrame(data)
        self.df = self.preprocessor.preprocess_df(self.df, 'prompt')

    def _train_model(self):
        X = self.df['clean_prompt'].fillna('')
        y = self.df['classification']

        self.vectorizer = TfidfVectorizer(max_features=17000, ngram_range=(1, 2))
        X_vec = self.vectorizer.fit_transform(X)

        self.model = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
        self.model.fit(X_vec, y)

        if self.model_output_dir:
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, self.model_output_dir / 'tfidf_vectorizer.joblib')
            joblib.dump(self.model, self.model_output_dir / 'linear_svm_model.joblib')
            print(f"Model and vectorizer saved to '{self.model_output_dir}'.")

    def classify_prompt(self, prompt: str) -> str:
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model is not trained. Please run the train() method first.")

        clean_prompt = self.preprocessor.preprocess(prompt)
        vectorized_prompt = self.vectorizer.transform([clean_prompt])
        prediction = self.model.predict(vectorized_prompt)
        return prediction[0]

    def train(self):
        if not self.json_file_path:
            raise ValueError("json_file_path must be provided for training.")
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df['clean_prompt'] = df['prompt'].apply(self.preprocessor.preprocess)

        X = df['clean_prompt'].fillna('')
        y = df['classification']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        vectorizer = TfidfVectorizer(max_features=17000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)

        model = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
        model.fit(X_train_vec, y_train)

        print("\n--- Performance on Validation Data ---")
        X_val_vec = vectorizer.transform(X_val)
        y_pred_val = model.predict(X_val_vec)
        print(classification_report(y_val, y_pred_val, zero_division=0))

        print("\n--- Performance on Unseen Test Data ---")
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        print(classification_report(y_test, y_pred))
        print("--------------------------------------------\n")

        self.vectorizer = TfidfVectorizer(max_features=17000, ngram_range=(1, 2))
        X_full_vec = self.vectorizer.fit_transform(X)

        self.model = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
        self.model.fit(X_full_vec, y)

        if self.model_output_dir:
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, self.model_output_dir / 'tfidf_vectorizer.joblib')
            joblib.dump(self.model, self.model_output_dir / 'linear_svm_model.joblib')
            print(f"Final model and vectorizer saved to '{self.model_output_dir}'.")

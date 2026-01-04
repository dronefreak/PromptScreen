import json
import pandas as pd
import string
import emoji
import nltk
from pathlib import Path
import joblib
import numpy as np

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

_ = nltk.download("wordnet")
_ = nltk.download("stopwords")
_ = nltk.download("averaged_perceptron_tagger_eng")
_ = nltk.download("punkt")
_ = nltk.download("punkt_tab")


class TextPreProcessor:
    def __init__(self, custom_stopwords=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

    def _get_wordnet_pos(self, word: str):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = emoji.demojize(text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = word_tokenize(text)
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                pos = self._get_wordnet_pos(token)
                lemma = self.lemmatizer.lemmatize(token, pos)
                filtered_tokens.append(lemma)

        return " ".join(filtered_tokens)

    def preprocess_df(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        df_copy = df.copy()
        output_column = "clean_prompt"
        df_copy[output_column] = df_copy[text_column].apply(self.preprocess)
        return df_copy


def length_complexity_features(texts):
    features = []
    attack_keywords = {"ignore", "system", "prompt", "act", "as", "instruction", "follow", "previous"}
    for text in texts:
        char_len = len(text)
        word_len = len(text.split())
        char_no_space = len(text.replace(' ', ''))

        words = text.split()
        if word_len > 0:
            avg_word_len = np.mean([len(w) for w in words])
            punct_ratio = text.count('.') / char_len if char_len > 0 else 0
            attack_density = sum(1 for w in words if w in attack_keywords) / word_len
            repetition_score = max([words.count(w) for w in set(words)]) / word_len if word_len > 0 else 0
        else:
            avg_word_len = 0
            punct_ratio = 0
            attack_density = 0
            repetition_score = 0

        features.append([
            char_len / 1000,
            word_len / 100,
            char_no_space / 1000,
            avg_word_len,
            punct_ratio,
            attack_density,
            repetition_score,
            1.0 / (1 + word_len)
        ])
    return np.array(features)


class JailbreakClassifier:
    # Added variant parameter for different configs
    def __init__(self, json_file_path: str, 
                 model_output_dir: str = None,
                 variant: str = "word_ngram_1_2"
                 ):
        self.json_file_path = json_file_path
        self.model_output_dir = Path(model_output_dir) if model_output_dir else None
        self.variant = variant
        self.preprocessor = TextPreProcessor()
        self.feature_union = None
        self.model = None
        self.df = None

    #new method for getting feature configs based on variant
    def _get_feature_config(self):
        """
        Variants explained:
        - baseline: Only word unigrams - captures individual words
        - word_bigram: Only 2-word sequences - "ignore previous"
        - word_trigram: Only 3-word sequences - "ignore previous instructions"
        - word_ngram_1_2: Unigrams + bigrams - captures both individual and paired words
        - word_ngram_1_3: Unigrams + bigrams + trigrams - most comprehensive word features
        - char_ngram_3_5: Character sequences 3-5 chars - "ign", "igno", "ignor"
        - char_ngram_2_4: Character sequences 2-4 chars - captures shorter patterns
        - hybrid_word_char: Combines word (1,2) and char (3,5) - both word and char features
        - hybrid_extended: Combines word (1,3) and char (2,5) - most comprehensive hybrid
        """
        configs = {
            "baseline": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(1, 1), analyzer='word'))
                ]
            },
            "word_bigram": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(2, 2), analyzer='word'))
                ]
            },
            "word_trigram": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(3, 3), analyzer='word'))
                ]
            },
            "word_ngram_1_2": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(1, 2), analyzer='word'))
                ]
            },
            "word_ngram_1_3": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(1, 3), analyzer='word'))
                ]
            },
            "char_ngram_3_5": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(3, 5), analyzer='char'))
                ]
            },
            "char_ngram_2_4": {
                "vectorizers": [
                    ('tfidf', TfidfVectorizer(max_features=17000, ngram_range=(2, 4), analyzer='char'))
                ]
            },
            "hybrid_word_char": {
                "vectorizers": [
                    ('tfidf_word', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), analyzer='word')),
                    ('tfidf_char', TfidfVectorizer(max_features=7000, ngram_range=(3, 5), analyzer='char'))
                ]
            },
            "hybrid_extended": {
                "vectorizers": [
                    ('tfidf_word', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), analyzer='word')),
                    ('tfidf_char', TfidfVectorizer(max_features=7000, ngram_range=(2, 5), analyzer='char'))
                ]
            }
        }
        return configs.get(self.variant, configs["word_ngram_1_2"])

    def _load_and_preprocess_data(self):
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.df = pd.DataFrame(data)
        self.df = self.preprocessor.preprocess_df(self.df, "prompt")

    def _train_model(self):
        X_text = self.df["clean_prompt"].fillna("")
        y = self.df["classification"]

        config = self._get_feature_config()
        vectorizers = config["vectorizers"]
        vectorizers.append(('length_features', FunctionTransformer(length_complexity_features)))
        
        self.feature_union = FeatureUnion(vectorizers)
        X_features = self.feature_union.fit_transform(X_text)
        self.model = LinearSVC(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        )
        self.model.fit(X_features, y)

        if self.model_output_dir:
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                self.feature_union, self.model_output_dir / f"feature_union_{self.variant}.joblib"
            )
            joblib.dump(self.model, self.model_output_dir / f"linear_svm_model_{self.variant}.joblib")
            print(f"Model variant '{self.variant}' saved to '{self.model_output_dir}'.")

    def classify_prompt(self, prompt: str) -> str:
        if not self.model or not self.feature_union:
            raise RuntimeError(
                "Model is not trained. Please run the train() method first."
            )

        clean_prompt = self.preprocessor.preprocess(prompt)
        features = self.feature_union.transform([clean_prompt])
        prediction = self.model.predict(features)
        return prediction[0]

    def train(self):
        if not self.json_file_path:
            raise ValueError("json_file_path must be provided for training.")
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        label_map = {
            "jailbreak": "jailbreak",
            "prompt-injection": "jailbreak",
            "benign": "benign",
        }
        df["classification"] = df["classification"].map(label_map)
        unknown_labels = set(df["classification"].unique()) - set(label_map.values())
        if unknown_labels:
            print("Warning: Unknown labels found:", unknown_labels)

        df["clean_prompt"] = df["prompt"].apply(self.preprocessor.preprocess)

        X = df["clean_prompt"].fillna("")
        y = df["classification"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        config = self._get_feature_config()
        vectorizers = config["vectorizers"]
        vectorizers.append(('length_features', FunctionTransformer(length_complexity_features)))
        
        feature_union = FeatureUnion(vectorizers)
        X_train_features = feature_union.fit_transform(X_train)
        model = LinearSVC(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        )
        model.fit(X_train_features, y_train)

        print(f"\n=== Training Variant: {self.variant} ===")
        print("\n--- Validation ---")
        X_val_features = feature_union.transform(X_val)
        y_pred_val = model.predict(X_val_features)
        print(classification_report(y_val, y_pred_val, zero_division=0))

        print("\n--- Unseen Test ---")
        X_test_features = feature_union.transform(X_test)
        y_pred = model.predict(X_test_features)
        print(classification_report(y_test, y_pred))
        print("--------------------------------------------\n")

        config_final = self._get_feature_config()
        vectorizers_final = config_final["vectorizers"]
        vectorizers_final.append(('length_features', FunctionTransformer(length_complexity_features)))
        
        self.feature_union = FeatureUnion(vectorizers_final)

        X_full_features = self.feature_union.fit_transform(X)
        self.model = LinearSVC(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        )
        self.model.fit(X_full_features, y)

        if self.model_output_dir:
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(
                self.feature_union, self.model_output_dir / f"feature_union_{self.variant}.joblib"
            )
            joblib.dump(self.model, self.model_output_dir / f"linear_svm_model_{self.variant}.joblib")
            print(f"Final model variant '{self.variant}' saved to '{self.model_output_dir}'.")

import joblib
from pathlib import Path
import emoji
import string
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

from typing import override
from defence.abstract_defence import AbstractDefence

class TextPreProcessor:
    def __init__(self, custom_stopwords=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

    def _get_wordnet_pos(self, word: str):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = emoji.demojize(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        filtered_tokens = [self.lemmatizer.lemmatize(token, self._get_wordnet_pos(token)) for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(filtered_tokens)


class JailbreakInferenceAPI(AbstractDefence):
    def __init__(self, model_dir: str):
        model_path = Path(model_dir) / 'linear_svm_model.joblib'
        vectorizer_path = Path(model_dir) / 'tfidf_vectorizer.joblib'

        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(f"Model or vectorizer not found in '{model_dir}'. Please run the training script first.")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = TextPreProcessor()

    @override
    def is_safe(self, query: str) -> bool:
        clean_prompt = self.preprocessor.preprocess(query)
        vectorized_prompt = self.vectorizer.transform([clean_prompt])
        prediction = self.model.predict(vectorized_prompt)
        return prediction[0] != 'jailbreak'

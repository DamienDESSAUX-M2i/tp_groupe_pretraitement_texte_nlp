import html
import re
import unicodedata
from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def ensure_stop_words_downloaded():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def ensure_punkt_tab_downloaded():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


ensure_stop_words_downloaded()
ensure_punkt_tab_downloaded()

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"
TRAIN_SET_SAMPLE_PATH = DATA_PATH / "train_sample.csv"
TEST_SET_SAMPLE_PATH = DATA_PATH / "test_sample.csv"
TRAIN_SET_SAMPLE_CLEAN_PATH = DATA_PATH / "train_sample_clean.csv"
TEST_SET_SAMPLE_CLEAN_PATH = DATA_PATH / "test_sample_clean.csv"


ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1F\x7F]")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z0-9#]+;")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NUMBER_PATTERN = re.compile(r"\d+")
CONTRACTIONS: dict[str, str] = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
}
CONTRACTION_PATTERN = re.compile(r"\b(" + "|".join(CONTRACTIONS.keys()) + r")\b")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
EN_STOPWORDS = set(stopwords.words("english"))
NEGATIVE_WORDS = {"not", "no", "never", "none", "don't", "didn't", "cannot", "won't"}
FINAL_STOPWORDS = EN_STOPWORDS - NEGATIVE_WORDS


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["review"])


def remove_short_reviews(df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    mask = df["review"].str.len() >= min_length
    return df[mask]


def special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    total_chars = len(text)
    special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    return special_chars / total_chars


def remove_high_special_char_reviews(
    df: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    mask = df["review"].apply(lambda x: special_char_ratio(x) <= threshold)
    return df[mask]


def ensure_clean_utf8(text: str, form: str = "NFD") -> str:
    text = unicodedata.normalize(form, text)
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = CONTROL_CHAR_PATTERN.sub("", text)
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review"] = df["review"].apply(ensure_clean_utf8)
    return df


def clean_review_text(text: str) -> str:
    text = HTML_TAG_PATTERN.sub("", text)
    text = html.unescape(text)
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = text.lower()
    text = CONTRACTION_PATTERN.sub(lambda match: CONTRACTIONS[match.group(0)], text)
    text = NUMBER_PATTERN.sub("", text)
    text = PUNCTUATION_PATTERN.sub("", text)
    return text.strip()


def clean_reviews_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review"] = df["review"].apply(clean_review_text)
    return df


def tokenize_words(text: str, lemmatize: bool = False, stem: bool = False) -> list[str]:
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if re.match(r"\w+", token)]
    tokens = [token for token in tokens if token not in FINAL_STOPWORDS]
    if stem:
        tokens = [stemmer.stem(token) for token in tokens]
    if lemmatize:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)


def tokenize_words_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review"] = df["review"].apply(lambda x: tokenize_words(x, True))
    return df


def clean_reviews_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_reviews(df)
    df = clean_reviews_column(df)
    df = remove_duplicates(df)
    df = remove_short_reviews(df)
    df = remove_high_special_char_reviews(df)
    df = tokenize_words_column(df)
    return df


def preprocess():
    train_df = pd.read_csv(TRAIN_SET_SAMPLE_PATH)
    train_df_clean = clean_reviews_pipeline(train_df)
    train_df_clean.to_csv(TRAIN_SET_SAMPLE_CLEAN_PATH, index=False)
    test_df = pd.read_csv(TEST_SET_SAMPLE_PATH)
    test_df_clean = clean_reviews_pipeline(test_df)
    test_df_clean.to_csv(TEST_SET_SAMPLE_CLEAN_PATH, index=False)


if __name__ == "__main__":
    preprocess()

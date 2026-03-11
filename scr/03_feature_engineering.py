import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import string
from textblob import TextBlob

# =============================
# PATHS
# =============================
PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"
TRAIN_SET_SAMPLE_PATH = DATA_PATH / "train_sample.csv"
TEST_SET_SAMPLE_PATH = DATA_PATH / "test_sample.csv"


# =============================
# DATA LOADING
# =============================
train_df = pd.read_csv(TRAIN_SET_SAMPLE_PATH)
test_df = pd.read_csv(TEST_SET_SAMPLE_PATH)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# =============================
# PHASE 3 : FEATURE ENGINEERING
# =============================

# --------------------------------
# 1. TF-IDF
# --------------------------------

vectorizer_tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = vectorizer_tfidf.fit_transform(train_df["review"])
X_test_tfidf = vectorizer_tfidf.transform(test_df["review"])

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)

# --------------------------------
# 2. Word2Vec
# --------------------------------

# Tokenisation
train_tokens = train_df["review"].apply(simple_preprocess)
test_tokens = test_df["review"].apply(simple_preprocess)

# Corpus complet (train + test pour meilleur vocabulaire)
corpus = pd.concat([train_tokens, test_tokens]).tolist()

print("Example tokens:", corpus[:2])

# Entraînement Word2Vec (Skip-gram)
model = Word2Vec(
    sentences=corpus,
    vector_size=300,
    window=5,
    min_count=2,
    sg=1,
    workers=4
)

print("Vocabulary size:", len(model.wv))

# --------------------------------
# 3. Document Embeddings
# --------------------------------

def document_vector(tokens):
    words = [w for w in tokens if w in model.wv]

    if len(words) == 0:
        return np.zeros(model.vector_size)

    return np.mean(model.wv[words], axis=0)


# Création des embeddings
X_train_w2v = np.array([document_vector(tokens) for tokens in train_tokens])
X_test_w2v = np.array([document_vector(tokens) for tokens in test_tokens])

print("Word2Vec train shape:", X_train_w2v.shape)
print("Word2Vec test shape:", X_test_w2v.shape)

# --------------------------------
# 4. Analyse des similarités
# --------------------------------

print("\nMost similar words to 'good':")
try:
    print(model.wv.most_similar("good", topn=10))
except KeyError:
    print("Word 'good' not in vocabulary")

print("\nSimilarity between 'good' and 'great':")
try:
    print(model.wv.similarity("good", "great"))
except KeyError:
    print("Words not found in vocabulary")
# --------------------------------
# . Features additionnelles (Longueur texte, Nombre mots uniques, Ratio ponctuation, Nombre majuscules, Sentiment lexicon score (TextBlob))
# --------------------------------
def extract_features(df):

    features = pd.DataFrame()

    features["text_length"] = df["review"].apply(len)
    features["num_unique_words"] = df["review"].apply(lambda x: len(set(x.split())))
    features["punctuation_ratio"] = df["review"].apply(
        lambda x: sum(c in string.punctuation for c in x) / max(len(x),1)
    )
    features["num_uppercase"] = df["review"].apply(
        lambda x: sum(1 for c in x if c.isupper())
    )
    features["sentiment"] = df["review"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    return features

X_train_extra = extract_features(train_df)
X_test_extra = extract_features(test_df)

print(X_train_extra.head())

# TF-IDF + features additionnelles
X_train_combined = hstack([X_train_tfidf, X_train_extra])
X_test_combined = hstack([X_test_tfidf, X_test_extra])

# 5. Analyse feature
#    - Quelle approche discrimine mieux?
#    - Features corrélées?
#    - Réduction dimension si besoin (PCA)

import seaborn as sns
import matplotlib.pyplot as plt

corr = X_train_extra.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.xticks(rotation=45, fontsize=10)
plt.title("Feature correlation matrix")

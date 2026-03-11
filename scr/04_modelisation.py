from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

RANDOM_STATE = 42
# --------------------------------
# Vectorizer from 03_feature_engineering.py
# --------------------------------
vectorizer_tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    lowercase=True
)

# --------------------------------
# data from 03_feature_engineering.py
# --------------------------------
PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"
TRAIN_SET_SAMPLE_PATH = DATA_PATH / "train_sample.csv"
TEST_SET_SAMPLE_PATH = DATA_PATH / "test_sample.csv"
train_df = pd.read_csv(TRAIN_SET_SAMPLE_PATH)
test_df = pd.read_csv(TEST_SET_SAMPLE_PATH)

# --------------------------------
# Logistic regression
# --------------------------------

pipeline_lr = Pipeline([
    ("tfidf", vectorizer_tfidf),
    ("clf", LogisticRegression(random_state=RANDOM_STATE))
])

param_grid_lr = {
    "clf__C": [0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "newton-cg"]
}

grid_lr = GridSearchCV(
    pipeline_lr,
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)

grid_lr.fit(train_df["review"], train_df["label"])
y_pred_lr = grid_lr.predict(test_df["review"])
y_proba_lr = grid_lr.predict_proba(test_df["review"])[:,1]

# --------------------------------
# SVM
# --------------------------------
pipeline_svm = Pipeline([
    ("tfidf", vectorizer_tfidf),
    ("clf", SVC(probability=True, random_state=RANDOM_STATE))
])

param_grid_svm = {
    "clf__C": [0.1, 1, 10],
    "clf__kernel": ["linear", "rbf"],
    "clf__gamma": ["scale", "auto"]
}

grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid=param_grid_svm,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
grid_svm.fit(train_df["review"], train_df["label"])
y_pred_svm = grid_svm.predict(test_df["review"])
y_proba_svm = grid_svm.predict_proba(test_df["review"])[:,1]

# --------------------------------
# Random Forest
# --------------------------------
pipeline_rf = Pipeline([
    ("tfidf", vectorizer_tfidf),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
])

param_grid_rf = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [10, 20, None],
    "clf__min_samples_split": [2, 5]
}

grid_rf = GridSearchCV(
    pipeline_rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
grid_rf.fit(train_df["review"], train_df["label"])
y_pred_rf = grid_rf.predict(test_df["review"])
y_proba_rf = grid_rf.predict_proba(test_df["review"])[:,1]

# --------------------------------
# Coting classifier
# --------------------------------
voting_pipeline = VotingClassifier(
    estimators=[
        ("lr", grid_lr.best_estimator_),
        ("svm", grid_svm.best_estimator_),
        ("rf", grid_rf.best_estimator_)
    ],
    voting="soft"
)

voting_pipeline.fit(train_df["review"], train_df["label"])
y_pred_vote = voting_pipeline.predict(test_df["review"])
y_proba_vote = voting_pipeline.predict_proba(test_df["review"])[:,1]

# --------------------------------
# Comparaison model
# --------------------------------
models = {
    "Logistic Regression": (y_pred_lr, y_proba_lr),
    "SVM": (y_pred_svm, y_proba_svm),
    "Random Forest": (y_pred_rf, y_proba_rf),
    "Voting": (y_pred_vote, y_proba_vote)
}

# --------------------------------
# Statistiques
# --------------------------------
import pandas as pd
results = []

for name, (pred, proba) in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(test_df["label"], pred),
        "Precision": precision_score(test_df["label"], pred),
        "Recall": recall_score(test_df["label"], pred),
        "F1": f1_score(test_df["label"], pred),
        "ROC_AUC": roc_auc_score(test_df["label"], proba)
    })

df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
print(df_results)
from pathlib import Path

import pandas as pd

PROJECT_PATH = Path(__file__).parent.parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"
TRAIN_SET_PATH = DATA_PATH / "train.ft.txt.bz2"
TRAIN_SET_SAMPLE_PATH = DATA_PATH / "train_sample.csv"
TEST_SET_PATH = DATA_PATH / "test.ft.txt.bz2"
TEST_SET_SAMPLE_PATH = DATA_PATH / "test_sample.csv"

N_ROWS = 1_000

train_df = pd.read_csv(
    TRAIN_SET_PATH,
    compression="bz2",
    header=None,
    sep="\t",
    engine="python",
    nrows=N_ROWS,
)

train_df[["label", "review"]] = train_df[0].str.split(" ", n=1, expand=True)
train_df["label"] = train_df["label"].str.replace("__label__", "").astype(int)
train_df = train_df[["review", "label"]]
train_df.to_csv(TRAIN_SET_SAMPLE_PATH, index=False)


test_df = pd.read_csv(
    TEST_SET_PATH,
    compression="bz2",
    header=None,
    sep="\t",
    engine="python",
    nrows=N_ROWS,
)

test_df[["label", "review"]] = test_df[0].str.split(" ", n=1, expand=True)
test_df["label"] = test_df["label"].str.replace("__label__", "").astype(int)
test_df = test_df[["review", "label"]]
test_df.to_csv(TEST_SET_SAMPLE_PATH, index=False)

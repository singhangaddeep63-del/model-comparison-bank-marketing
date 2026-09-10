"""Data loading and preprocessing for the Bank Marketing dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RAW_PATH = "data/bank-full.csv"

NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
TARGET = "y"

# duration is known only after the call happens, so it leaks the outcome.
# Dropped for a realistic pre-call predictive model (noted in README).
LEAKY_FEATURE = "duration"


def load_data(path: str = RAW_PATH, drop_leaky: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})
    if drop_leaky and LEAKY_FEATURE in df.columns:
        df = df.drop(columns=[LEAKY_FEATURE])
    return df


def get_feature_lists(drop_leaky: bool = True):
    numeric = [f for f in NUMERIC_FEATURES if not (drop_leaky and f == LEAKY_FEATURE)]
    return numeric, CATEGORICAL_FEATURES


def build_preprocessor(drop_leaky: bool = True) -> ColumnTransformer:
    numeric, categorical = get_feature_lists(drop_leaky)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    numeric, categorical = get_feature_lists()
    X = df[numeric + categorical]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

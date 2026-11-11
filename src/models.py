"""Model definitions. Every model outputs probabilities for [Home, Draw, Away]."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

ALL_MODELS = ["logreg", "rf", "hgb", "xgb", "poisson"]
LABELS = {
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "hgb": "Gradient boosting (sklearn)",
    "xgb": "XGBoost",
    "poisson": "Poisson goals model",
}


def xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception as exc:      # not installed, OR installed but libomp is missing (Mac)
        if not getattr(xgb_available, "_warned", False):
            print("  XGBoost unavailable (on a Mac run `brew install libomp`); "
                  f"continuing without it. [{type(exc).__name__}]")
            xgb_available._warned = True
        return False


XGB_PARAMS = dict(
    objective="multi:softprob", n_estimators=1000, learning_rate=0.03, max_depth=3,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
    eval_metric="mlogloss", tree_method="hist", random_state=config.RANDOM_STATE,
    n_jobs=-1,
)



class PoissonGoals:
    """Predicts expected goals for each side (Poisson regression).

    Needed for knockout ties: aggregate scores, extra time and penalties require
    scorelines, not just win/draw/loss probabilities. It also yields H/D/A probabilities
    (independent Poisson goals), so it is evaluated with the same metrics as the others.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        mk = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler()),
                               ("reg", PoissonRegressor(alpha=alpha, max_iter=2000))])
        self.home, self.away = mk(), mk()

    def fit(self, X, hg, ag):
        self.home.fit(X, hg)
        self.away.fit(X, ag)
        return self

    def lambdas(self, X):
        return self.home.predict(X), self.away.predict(X)

    def predict_proba(self, X, max_goals: int = 10) -> np.ndarray:
        lh, la = self.lambdas(X)
        ks = np.arange(max_goals + 1)
        fact = np.cumprod(np.maximum(ks, 1)).astype(float)
        ph = np.exp(-lh)[:, None] * lh[:, None] ** ks / fact
        pa = np.exp(-la)[:, None] * la[:, None] ** ks / fact
        grid = ph[:, :, None] * pa[:, None, :]
        home_w = (ks[:, None] > ks[None, :])
        draw = np.eye(len(ks), dtype=bool)
        P = np.stack([(grid * home_w).sum((1, 2)), (grid * draw).sum((1, 2)),
                      (grid * ~(home_w | draw)).sum((1, 2))], axis=1)
        return P / P.sum(axis=1, keepdims=True)


def _make(name: str):
    if name == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.5, max_iter=2000))])
    if name == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(
                             n_estimators=500, min_samples_leaf=10, max_features="sqrt",
                             n_jobs=-1, random_state=config.RANDOM_STATE))])
    if name == "hgb":   # handles missing values natively
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=250, l2_regularization=5.0,
            min_samples_leaf=20, random_state=config.RANDOM_STATE)
    raise ValueError(f"unknown model {name}")


def fit_model(name: str, train: pd.DataFrame, features: list[str],
              val_start: str | None = None):
    """Fit one model on `train`.

    For XGBoost, the last part of `train` (from `val_start`) is used only to choose the
    number of trees with early stopping; the final model is then refit on all of `train`.
    """
    X, y = train[features], train["y"].astype(int)
    if name == "poisson":
        return PoissonGoals().fit(X, train["FTHG"].astype(float), train["FTAG"].astype(float))
    if name != "xgb":
        return _make(name).fit(X, y)

    if not xgb_available():
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    from xgboost import XGBClassifier

    n_trees = XGB_PARAMS["n_estimators"]
    if val_start is not None:
        fit_part = train[train["Date"] < val_start]
        val_part = train[train["Date"] >= val_start]
        probe = XGBClassifier(**XGB_PARAMS, early_stopping_rounds=50)
        probe.fit(fit_part[features], fit_part["y"].astype(int),
                  eval_set=[(val_part[features], val_part["y"].astype(int))], verbose=False)
        n_trees = int(probe.best_iteration) + 1
        print(f"    XGBoost early stopping chose {n_trees} trees")
    final = XGBClassifier(**{**XGB_PARAMS, "n_estimators": n_trees})
    return final.fit(X, y)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))


def importances(model, name: str, features: list[str]):
    """Return (feature_names, importance_values) or None."""
    if name == "xgb":
        return features, np.asarray(model.feature_importances_)
    if name == "rf":
        return features, np.asarray(model.named_steps["clf"].feature_importances_)
    if name == "logreg":
        return features, np.abs(model.named_steps["clf"].coef_).mean(axis=0)
    return None

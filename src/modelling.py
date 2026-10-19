"""Reusable helpers for the Climate Health challenge: data loading, grouped CV, experiments and submissions."""

import itertools
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
TARGET = "is_climate_sensitive"


def set_seeds(seed=SEED):
    """Fix random seeds everywhere (the rules require reproducible results)."""
    random.seed(seed)
    np.random.seed(seed)


# ---------- Data ----------

def load_data(data_dir="../data/raw"):
    """Load train, test and the sample submission, join the climate features and add 0/1 zone/gender columns."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "Train.csv", parse_dates=["deathdate"])
    test = pd.read_csv(data_dir / "Test.csv", parse_dates=["deathdate"])
    sample_submission = pd.read_csv(data_dir / "SampleSubmission.csv")

    # hot_days_30d is always 0, and deathdate is already in train/test
    climate = pd.read_csv(data_dir / "climate_features.csv")
    climate = climate.drop(columns=["deathdate", "hot_days_30d"])
    train = train.merge(climate, on="ID", how="left", validate="one_to_one")
    test = test.merge(climate, on="ID", how="left", validate="one_to_one")

    for df in [train, test]:
        df["is_rural"] = (df["zone"] == "Rural").astype(int)
        df["is_male"] = (df["gender"] == "Male").astype(int)
        df["year"] = df["deathdate"].dt.year

    return train, test, sample_submission


def location_groups(train):
    """CV groups = location, with two names merged that are really the same place."""
    groups = train["location"].replace({"Buwoya, Entebbe City, Uganda": "Buwaya, Entebbe City, Uganda"})

    # The two "Brac Uganda Iganga Regional Branch, ..." names have identical coordinates
    is_brac = groups.str.startswith("Brac Uganda Iganga Regional Branch")
    groups = groups.copy()
    groups[is_brac] = "Brac Uganda Iganga Regional Branch"
    return groups


# ---------- Models ----------

def make_model(kind, params, seed=SEED):
    """Build a model from a name and a dict of hyperparameters. The seed is always added."""
    if kind == "histgb":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    if kind == "catboost":
        return CatBoostClassifier(**params, random_seed=seed, verbose=0, allow_writing_files=False)
    if kind == "lightgbm":  # needs `brew install libomp` on macOS
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, random_state=seed, verbose=-1)
    if kind == "tabpfn":  # needs `pip install tabpfn`
        return make_tabpfn_v2(params, seed)
    raise ValueError(f"Unknown model kind: {kind}")


def make_tabpfn_v2(params, seed):
    """TabPFN loaded with the v2 weights explicitly.

    Newer TabPFN releases default to weights under a non-commercial licence, which conflicts with
    assigning copyright to Zindi, so the v2 weights are requested by name.
    """
    from tabpfn import TabPFNClassifier
    try:
        from tabpfn.constants import ModelVersion
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "This TabPFN version has no `create_default_for_version(ModelVersion.V2)`. Check the TabPFN "
            "README for how to load the v2 weights in your version, or pin an older release that has it."
        ) from error
    model.set_params(**params, random_state=seed)
    return model


class AveragedModel(ClassifierMixin, BaseEstimator):
    """Averages the predicted probabilities of several models. Behaves like one model (fit / predict_proba)."""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        self.fitted_models_ = []
        for model in self.models:
            fitted = clone(model).fit(X, y)
            self.fitted_models_.append(fitted)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        # Equal weights unless weights were given
        if self.weights is None:
            weights = [1 / len(self.fitted_models_)] * len(self.fitted_models_)
        else:
            weights = self.weights

        prob = 0
        for model, weight in zip(self.fitted_models_, weights):
            prob = prob + weight * model.predict_proba(X)[:, 1]
        return np.column_stack([1 - prob, prob])


class ColumnModel(ClassifierMixin, BaseEstimator):
    """A model that only uses its own list of columns, so models with different features can be combined."""

    def __init__(self, model, features):
        self.model = model
        self.features = features

    def fit(self, X, y):
        self.fitted_model_ = clone(self.model).fit(X[self.features], y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.fitted_model_.predict_proba(X[self.features])


class AgeSplitModel(ClassifierMixin, BaseEstimator):
    """Main model for everyone under `age_cut`, a separate model (trained on adults only) for everyone else.

    The main model is trained on all rows; the adult model only on rows with age >= age_cut.
    Both should be ColumnModels, so each uses its own features. X must include an "age" column.
    """

    def __init__(self, main_model, adult_model, age_cut=5):
        self.main_model = main_model
        self.adult_model = adult_model
        self.age_cut = age_cut

    def fit(self, X, y):
        adult = (X["age"] >= self.age_cut).values
        self.fitted_main_ = clone(self.main_model).fit(X, y)
        self.fitted_adult_ = clone(self.adult_model).fit(X[adult], y[adult])
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        adult = (X["age"] >= self.age_cut).values
        prob = self.fitted_main_.predict_proba(X)[:, 1]
        if adult.any():
            prob[adult] = self.fitted_adult_.predict_proba(X[adult])[:, 1]
        return np.column_stack([1 - prob, prob])


def seed_average(kind, params, seeds=(42, 43, 44)):
    """The same model trained with several seeds and averaged.

    Seeds only make a difference if the model has randomness, e.g. column or row subsampling.
    """
    models = []
    for seed in seeds:
        models.append(make_model(kind, params, seed=seed))
    return AveragedModel(models)


# ---------- Scoring ----------

def competition_score(y_true, prob):
    """0.6 x F1 (at the fixed 0.5 threshold) + 0.4 x ROC-AUC, like the leaderboard."""
    f1 = f1_score(y_true, (prob >= 0.5).astype(int))
    auc = roc_auc_score(y_true, prob)
    return {"score": 0.6 * f1 + 0.4 * auc, "f1": f1, "auc": auc}


# ---------- Experiments ----------

class Experiments:
    """Runs and records experiments with grouped CV, repeated over several random fold splits.

    Every experiment uses the same splits, so their scores can be compared directly.
    """

    def __init__(self, train, n_splits=5, split_seeds=(0, 1, 2, 3, 4)):
        self.train = train
        self.y = train[TARGET]
        groups = location_groups(train)

        # One list of (train_idx, valid_idx) folds per split seed
        self.splits = {}
        for split_seed in split_seeds:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
            self.splits[split_seed] = list(splitter.split(train, self.y, groups))

        self.oofs = {}     # name -> {split_seed: out-of-fold probabilities}
        self.results = {}  # name -> scores

    def out_of_fold(self, model, features):
        """Predictions for every train row, each made by a model that never saw that row's village."""
        X = self.train[features]
        oofs = {}
        for split_seed, folds in self.splits.items():
            oof = np.zeros(len(X))
            for train_idx, valid_idx in folds:
                fold_model = clone(model).fit(X.iloc[train_idx], self.y.iloc[train_idx])
                oof[valid_idx] = fold_model.predict_proba(X.iloc[valid_idx])[:, 1]
            oofs[split_seed] = oof
        return oofs

    def split_scores(self, oofs):
        """Mean fold score for each split."""
        scores = []
        for split_seed, folds in self.splits.items():
            fold_scores = []
            for _, valid_idx in folds:
                fold_score = competition_score(self.y.iloc[valid_idx], oofs[split_seed][valid_idx])
                fold_scores.append(fold_score["score"])
            scores.append(np.mean(fold_scores))
        return scores

    @staticmethod
    def pos_rate(oofs):
        """Share of rows predicted positive (probability >= 0.5), averaged over the splits."""
        rates = []
        for oof in oofs.values():
            rates.append((oof >= 0.5).mean())
        return float(np.mean(rates))

    def blend_oofs(self, names, weights):
        """Weighted average of the out-of-fold predictions of experiments that were already run."""
        blended = {}
        for split_seed in self.splits:
            prob = 0
            for name, weight in zip(names, weights):
                prob = prob + weight * self.oofs[name][split_seed]
            blended[split_seed] = prob
        return blended

    def record(self, name, oofs):
        """Score out-of-fold predictions and store them under `name`."""
        scores = self.split_scores(oofs)
        self.oofs[name] = oofs

        result = {"mean": np.mean(scores), "worst_split": np.min(scores)}
        for split_seed, score in zip(self.splits, scores):
            result[f"split_{split_seed}"] = score
        self.results[name] = result
        print(f"{name}: {np.mean(scores):.4f} (worst split {np.min(scores):.4f})")

    def run(self, name, model, features):
        """Score one model on one feature list."""
        oofs = self.out_of_fold(model, features)
        self.record(name, oofs)

    def blend(self, name, names, weights=None):
        """Score a weighted average of experiments that were already run (equal weights by default)."""
        if weights is None:
            weights = [1 / len(names)] * len(names)
        self.record(name, self.blend_oofs(names, weights))

    def grid_search(self, kind, grid, features):
        """Try every combination in `grid` (a dict of lists). Returns a table sorted by mean score.

        Results are not added to the experiment table; run the winning settings with `run` afterwards.
        """
        rows = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            oofs = self.out_of_fold(make_model(kind, params), features)
            scores = self.split_scores(oofs)

            row = dict(params)
            row["mean"] = np.mean(scores)
            row["worst_split"] = np.min(scores)
            rows.append(row)

        table = pd.DataFrame(rows)
        table = table.sort_values("mean", ascending=False)
        return table.reset_index(drop=True)

    def random_search(self, kind, space, features, n_iter=80, seed=SEED):
        """Try `n_iter` random combinations from `space` (a dict of lists). Returns a table sorted by mean score.

        Like grid_search, results are not added to the experiment table. The "params" column holds each
        setting's exact dict, ready to pass to make_model.
        """
        all_combinations = list(itertools.product(*space.values()))
        n_picks = min(n_iter, len(all_combinations))
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(all_combinations), size=n_picks, replace=False)

        rows = []
        for i, pick in enumerate(picks, start=1):
            params = dict(zip(space.keys(), all_combinations[pick]))
            oofs = self.out_of_fold(make_model(kind, params), features)
            scores = self.split_scores(oofs)

            row = dict(params)
            row["mean"] = np.mean(scores)
            row["worst_split"] = np.min(scores)
            row["pred_pos_rate"] = self.pos_rate(oofs)
            row["params"] = params
            rows.append(row)
            print(f"Random search: {i}/{n_picks}", end="\r")
        print()

        table = pd.DataFrame(rows)
        table = table.sort_values("mean", ascending=False)
        return table.reset_index(drop=True)

    def weight_search(self, names, step=0.1):
        """Score every blend of the named experiments with weights in steps of `step` that add up to 1.

        Returns a table sorted by mean score. Nothing is added to the experiment table.
        """
        n_steps = round(1 / step)
        rows = []
        for counts in itertools.product(range(n_steps + 1), repeat=len(names)):
            if sum(counts) != n_steps:
                continue  # weights must add up to 1

            weights = []
            for count in counts:
                weights.append(count / n_steps)
            blended = self.blend_oofs(names, weights)
            scores = self.split_scores(blended)

            row = {}
            for name, weight in zip(names, weights):
                row[f"w_{name}"] = weight
            row["mean"] = np.mean(scores)
            row["worst_split"] = np.min(scores)
            row["pred_pos_rate"] = self.pos_rate(blended)
            rows.append(row)

        table = pd.DataFrame(rows)
        table = table.sort_values("mean", ascending=False)
        return table.reset_index(drop=True)

    def feature_importance(self, model, features, n_repeats=5):
        """How much the AUC drops when each column is shuffled in the held-out villages, averaged over all folds.

        Near zero or negative means the model doesn't really use that column for new villages.
        """
        X = self.train[features]
        drops = []
        for folds in self.splits.values():
            for train_idx, valid_idx in folds:
                fold_model = clone(model).fit(X.iloc[train_idx], self.y.iloc[train_idx])
                result = permutation_importance(fold_model, X.iloc[valid_idx], self.y.iloc[valid_idx],
                                                scoring="roc_auc", n_repeats=n_repeats, random_state=SEED)
                drops.append(result.importances_mean)

        importance = pd.Series(np.mean(drops, axis=0), index=features)
        return importance.sort_values(ascending=False)

    def table(self, baseline):
        """All experiments, with the gain over `baseline` and how many splits each one wins."""
        board = pd.DataFrame(self.results).T
        board = board.sort_values("mean", ascending=False)

        split_columns = []
        for split_seed in self.splits:
            split_columns.append(f"split_{split_seed}")

        board["gain_vs_baseline"] = board["mean"] - board.loc[baseline, "mean"]
        beats_baseline = board[split_columns] > board.loc[baseline, split_columns]
        board["splits_beating_baseline"] = beats_baseline.sum(axis=1)
        return board

    def compare(self, names, baseline, title):
        """Compare experiments against a baseline: draws a bar chart and returns a results table.

        Table columns: mean CV score, worst split, gain over the baseline, number of splits that beat the
        baseline, and the predicted-positive rate (share of rows at or above 0.5).
        """
        # Baseline first, then the other names, each only once
        rows_to_show = [baseline]
        for name in names:
            if name not in rows_to_show:
                rows_to_show.append(name)

        board = self.table(baseline).loc[rows_to_show]
        positive_rates = []
        for name in rows_to_show:
            positive_rates.append(self.pos_rate(self.oofs[name]))
        board["pred_pos_rate"] = positive_rates
        board = board.sort_values("mean", ascending=False)

        # Bar chart: best at the top, baseline in red
        names_top_to_bottom = list(board.index)
        names_bottom_to_top = names_top_to_bottom[::-1]
        scores_bottom_to_top = board["mean"].values[::-1]
        colors = []
        for name in names_bottom_to_top:
            if name == baseline:
                colors.append("indianred")
            else:
                colors.append("steelblue")

        plt.figure(figsize=(9, 0.45 * len(board) + 1.5))
        plt.barh(names_bottom_to_top, scores_bottom_to_top, color=colors)
        plt.axvline(board.loc[baseline, "mean"], color="indianred", linestyle="--", linewidth=1,
                    label=f"baseline: {baseline}")
        plt.xlim(board["mean"].min() - 0.003, board["mean"].max() + 0.003)
        plt.xlabel("Mean CV score (0.6 × F1 + 0.4 × AUC)")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        columns = ["mean", "worst_split", "gain_vs_baseline", "splits_beating_baseline", "pred_pos_rate"]
        return board[columns].round(4)

    def plot(self, baseline):
        """Bar chart of every experiment (used by the older modelling.ipynb)."""
        board = self.table(baseline)
        plt.figure(figsize=(9, 0.5 * len(board) + 1.5))
        plt.barh(board.index[::-1], board["mean"][::-1], color="steelblue")
        plt.axvline(board.loc[baseline, "mean"], color="red", linestyle="--", label=baseline)
        plt.xlim(board["mean"].min() - 0.005, board["mean"].max() + 0.005)
        plt.xlabel("Mean CV score over all splits")
        plt.legend()
        plt.show()


# ---------- Submission ----------

def make_submission(model, train, test, features, sample_submission, path):
    """Train on all of train, predict test, check the file against the rules, and save it."""
    final_model = clone(model).fit(train[features], train[TARGET])
    prob = final_model.predict_proba(test[features])[:, 1]

    predictions = pd.DataFrame({"ID": test["ID"], "TargetF1": (prob >= 0.5).astype(int), "TargetRAUC": prob})
    submission = sample_submission[["ID"]].merge(predictions, on="ID", how="left")

    assert len(submission) == len(sample_submission), "row count doesn't match the sample submission"
    assert list(submission.columns) == list(sample_submission.columns), "columns don't match the sample submission"
    assert submission.notna().all().all(), "some IDs have no prediction"
    assert submission["TargetRAUC"].between(0, 1).all(), "probabilities outside 0-1"

    submission.to_csv(path, index=False)
    print(f"Saved {path} | share predicted climate-sensitive: {submission['TargetF1'].mean():.3f} "
          f"(train rate: {train[TARGET].mean():.3f})")
    return submission

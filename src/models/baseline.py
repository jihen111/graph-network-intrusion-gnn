"""Flow-only classical baseline."""
from sklearn.ensemble import RandomForestClassifier


def make_baseline(seed: int, trees: int = 120) -> RandomForestClassifier:
    """A balanced random forest is the no-graph reference model."""
    return RandomForestClassifier(
        n_estimators=trees,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=4,
        random_state=seed,
    )

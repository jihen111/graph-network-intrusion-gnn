"""Flow-only classical baseline (no graph)."""
from sklearn.ensemble import RandomForestClassifier


def make_baseline(seed: int, trees: int = 100, max_samples: float | None = 0.3) -> RandomForestClassifier:
    """Balanced random forest on the same engineered flow features as the GNN."""
    return RandomForestClassifier(
        n_estimators=trees, class_weight="balanced_subsample", min_samples_leaf=2,
        max_features="sqrt", max_samples=max_samples, n_jobs=-1, random_state=seed,
    )

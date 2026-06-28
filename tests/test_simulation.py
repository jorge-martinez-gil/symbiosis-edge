import pandas as pd

from symbiosis_edge import (
    DATASET_PRESETS,
    METHODS,
    SimParams,
    default_datasets,
    simulate_datasets,
    simulate_one_run,
)

SMALL = SimParams(n=400, drift_t=150)


def test_determinism_same_seed():
    a = simulate_one_run(dataset="SYNTHETIC", seed=7, params=SMALL)
    b = simulate_one_run(dataset="SYNTHETIC", seed=7, params=SMALL)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = simulate_one_run(dataset="SYNTHETIC", seed=1, params=SMALL)
    b = simulate_one_run(dataset="SYNTHETIC", seed=2, params=SMALL)
    assert not a.equals(b)


def test_schema_and_shape():
    df = simulate_one_run(dataset="SYNTHETIC", seed=0, params=SMALL)
    expected_cols = {
        "dataset", "t", "method", "y_true", "y_pred",
        "q_oracle", "q_human", "oracle_correct", "human_correct",
    }
    assert expected_cols.issubset(df.columns)
    assert set(df["method"].unique()) == set(METHODS)
    assert len(df) == SMALL.n * len(METHODS)
    assert df["t"].min() == 0 and df["t"].max() == SMALL.n - 1


def test_static_never_queries():
    df = simulate_one_run(dataset="SYNTHETIC", seed=0, params=SMALL)
    static = df[df["method"] == "Static"]
    assert not static["q_oracle"].any()
    assert not static["q_human"].any()


def test_symbiosis_query_rate_within_budget():
    params = SimParams(n=1500, drift_t=300, b_oracle=0.12, b_human=0.05)
    df = simulate_one_run(dataset="SYNTHETIC", seed=0, params=params)
    sym = df[df["method"] == "Symbiosis-Edge"]
    rate = (sym["q_oracle"] | sym["q_human"]).mean()
    # Realised escalation should be in the neighbourhood of b_oracle + b_human.
    assert rate < 0.30


def test_simulate_datasets_adds_seed_and_concats():
    df = simulate_datasets(
        datasets=default_datasets(),
        params_by_dataset={k: SMALL for k in DATASET_PRESETS},
        seeds=[0, 1],
    )
    assert "seed" in df.columns
    assert set(df["seed"].unique()) == {0, 1}
    assert set(df["dataset"].unique()) == {"SECOM", "APS", "SYNTHETIC"}
    assert len(df) == SMALL.n * len(METHODS) * 3 * 2


def test_labels_in_range():
    df = simulate_one_run(dataset="SYNTHETIC", seed=3, params=SMALL)
    assert df["y_true"].between(0, SMALL.k_classes - 1).all()
    assert df["y_pred"].between(0, SMALL.k_classes - 1).all()

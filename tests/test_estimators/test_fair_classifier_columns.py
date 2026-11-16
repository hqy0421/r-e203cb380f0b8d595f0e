"""Tests for the column bookkeeping of the fairness classifiers.

These tests check that the sensitive column, the design matrix that is used for prediction and `coef_` all refer to
the same original columns, for pandas/Polars dataframes as well as for plain arrays.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklego.linear_model import DemographicParityClassifier, EqualOpportunityClassifier

pytestmark = pytest.mark.cvxpy


@pytest.fixture
def data():
    """A small binary problem where the sensitive attribute is informative."""
    rng = np.random.default_rng(7)
    n = 300
    sensitive = rng.integers(0, 2, n).astype(float)
    noise_1 = rng.normal(size=n)
    noise_2 = rng.normal(size=n)
    logit = 2.0 * sensitive + 0.5 * noise_1 + 0.8 * noise_2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x1": sensitive, "x2": noise_1, "x3": noise_2}), y


DATAFRAME_MAKERS = {
    "pandas": lambda df: df,
    "polars": pl.from_pandas,
}
MULTICLASS = ["ovr", "ovo"]


def _classifier(cls, multi_class, **kwargs):
    extra = {"positive_target": 1} if cls is EqualOpportunityClassifier else {}
    return cls(covariance_threshold=0.1, penalty=None, multi_class=multi_class, **extra, **kwargs)


@pytest.mark.parametrize("make", DATAFRAME_MAKERS.values(), ids=DATAFRAME_MAKERS.keys())
@pytest.mark.parametrize("multi_class", MULTICLASS)
@pytest.mark.parametrize("cls", [DemographicParityClassifier, EqualOpportunityClassifier])
def test_reordered_dataframe_predicts_identically(cls, multi_class, make, data):
    """Predictions must not depend on the physical column order of a dataframe."""
    df, y = data
    X = make(df)

    model = _classifier(cls, multi_class, sensitive_cols="x1").fit(X, y)
    reordered = X[["x3", "x1", "x2"]] if make is DATAFRAME_MAKERS["pandas"] else X.select(["x3", "x1", "x2"])

    np.testing.assert_array_equal(model.predict(X), model.predict(reordered))
    np.testing.assert_allclose(model.decision_function(X), model.decision_function(reordered))
    assert model.score(X, y) == pytest.approx(model.score(reordered, y))
    assert model.predict(X[:1]).shape == (1,)
    assert model.decision_function(X[:1]).shape == (1,)


@pytest.mark.parametrize("make", DATAFRAME_MAKERS.values(), ids=DATAFRAME_MAKERS.keys())
@pytest.mark.parametrize(
    "cls, extra", [(DemographicParityClassifier, {}), (EqualOpportunityClassifier, {"positive_target": 1})]
)
def test_reordered_dataframe_probabilities(cls, extra, make, data):
    """`predict_proba` shares the same design matrix as `decision_function`."""
    df, y = data
    X = make(df)

    model = cls(covariance_threshold=0.1, penalty=None, sensitive_cols="x1", **extra).fit(X, y)
    reordered = X[["x2", "x3", "x1"]] if make is DATAFRAME_MAKERS["pandas"] else X.select(["x2", "x3", "x1"])

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba, model.predict_proba(reordered))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert model.predict_proba(X[:1]).shape == (1, 2)

    # probabilities and scores agree
    decision = model.decision_function(reordered)
    np.testing.assert_allclose(proba[:, 1], 1 / (1 + np.exp(-decision)))
    np.testing.assert_array_equal(model.predict(reordered), model.classes_[(decision > 0).astype(int)])


@pytest.mark.parametrize("make", DATAFRAME_MAKERS.values(), ids=DATAFRAME_MAKERS.keys())
@pytest.mark.parametrize("multi_class", MULTICLASS)
@pytest.mark.parametrize("train_sensitive_cols", [False, True])
def test_coef_names_match_design_matrix(train_sensitive_cols, multi_class, make, data):
    """`coef_names_`, the design matrix and the shape of `coef_` describe the same columns."""
    df, y = data
    X = make(df)

    model = _classifier(
        DemographicParityClassifier, multi_class, sensitive_cols="x1", train_sensitive_cols=train_sensitive_cols
    ).fit(X, y)
    inner = model.estimators_[0]

    expected_names = ["x1", "x2", "x3"] if train_sensitive_cols else ["x2", "x3"]
    assert inner.coef_names_ == expected_names
    assert model.coef_names_ == expected_names
    assert inner.coef_.shape == (1, len(expected_names))
    assert inner.sensitive_col_idx_ == [0]
    assert inner.sensitive_cols_ == ["x1"]
    # the intercept lives in `intercept_`, never in `coef_`
    assert inner.intercept_.shape == (1,)
    assert model.n_features_in_ == 3
    assert list(model.feature_names_in_) == ["x1", "x2", "x3"]


@pytest.mark.parametrize("train_sensitive_cols", [False, True])
def test_sensitive_column_always_constrains_even_when_not_a_predictor(train_sensitive_cols, data):
    """The constraint reads the original sensitive values for both `train_sensitive_cols` settings."""
    df, y = data

    def fit_with(threshold):
        return DemographicParityClassifier(
            covariance_threshold=threshold,
            penalty=None,
            sensitive_cols="x1",
            train_sensitive_cols=train_sensitive_cols,
        ).fit(df, y)

    free = fit_with(None)
    constrained = fit_with(1e-4)

    # A vacuous (silently dropped) constraint would leave the coefficients untouched.
    assert not np.allclose(free.estimators_[0].coef_, constrained.estimators_[0].coef_, atol=1e-3)

    # and the constraint itself is what changed the solution: the covariance is small for the constrained model
    design = constrained.estimators_[0]
    X_design = df[design.coef_names_].to_numpy()
    decision = X_design @ design.coef_.ravel() + design.intercept_[0]
    sensitive = df["x1"].to_numpy()
    covariance = np.mean((sensitive - sensitive.mean()) * decision)
    assert abs(covariance) < 1e-2


@pytest.mark.parametrize("multi_class", MULTICLASS)
def test_array_input_uses_positions(multi_class, data):
    """Array input is interpreted positionally and reports positional names."""
    df, y = data
    X = df.to_numpy()

    model = _classifier(DemographicParityClassifier, multi_class, sensitive_cols=0).fit(X, y)
    inner = model.estimators_[0]

    assert inner.sensitive_col_idx_ == [0]
    assert inner.sensitive_cols_ == [0]
    assert model.coef_names_ == [1, 2]
    assert not hasattr(model, "feature_names_in_")

    # same result as the dataframe fit with the same positional declaration
    df_model = _classifier(DemographicParityClassifier, multi_class, sensitive_cols="x1").fit(df, y)
    np.testing.assert_allclose(model.decision_function(X), df_model.decision_function(df))
    np.testing.assert_allclose(
        np.sort(model.estimators_[0].coef_.ravel()), np.sort(df_model.estimators_[0].coef_.ravel())
    )


@pytest.mark.parametrize("multi_class", MULTICLASS)
def test_dataframe_sensitive_col_by_position(multi_class, data):
    """An integer declaration on a dataframe still refers to the original position."""
    df, y = data

    model = _classifier(DemographicParityClassifier, multi_class, sensitive_cols=0).fit(df, y)
    assert model.estimators_[0].sensitive_cols_ == ["x1"]
    assert model.estimators_[0].sensitive_col_idx_ == [0]


@pytest.mark.parametrize(
    "sensitive_cols, message",
    [
        (None, "must be specified"),
        ([], "is empty"),
        (["x1", "x2"], "exactly one sensitive column"),
        ("missing", "was not found"),
        (7, "out of range"),
        (True, "column name or index"),
    ],
)
def test_invalid_sensitive_declaration_on_dataframe_raises(sensitive_cols, message, data):
    """A wrong declaration must raise instead of fitting an unconstrained model."""
    df, y = data

    with pytest.raises(ValueError, match=message):
        _classifier(DemographicParityClassifier, "ovr", sensitive_cols=sensitive_cols).fit(df, y)


@pytest.mark.parametrize(
    "sensitive_cols, message",
    [
        (None, "must be specified"),
        ("x1", "column index"),
        ([0, 1], "exactly one sensitive column"),
        (7, "out of range"),
    ],
)
def test_invalid_sensitive_declaration_on_array_raises(sensitive_cols, message, data):
    """Column names cannot be used with array input, and indices must be in range."""
    df, y = data

    with pytest.raises(ValueError, match=message):
        _classifier(DemographicParityClassifier, "ovr", sensitive_cols=sensitive_cols).fit(df.to_numpy(), y)


@pytest.mark.parametrize("make", DATAFRAME_MAKERS.values(), ids=DATAFRAME_MAKERS.keys())
@pytest.mark.parametrize("multi_class", MULTICLASS)
def test_missing_or_extra_prediction_column_raises(multi_class, make, data):
    """Prediction must fail loudly when the columns do not match the fitted schema."""
    df, y = data
    X = make(df)
    model = _classifier(DemographicParityClassifier, multi_class, sensitive_cols="x1").fit(X, y)

    missing = X[["x1", "x2"]] if make is DATAFRAME_MAKERS["pandas"] else X.select(["x1", "x2"])
    extra = X.assign(x4=1.0) if make is DATAFRAME_MAKERS["pandas"] else X.with_columns(pl.lit(1.0).alias("x4"))

    with pytest.raises(ValueError, match="feature names"):
        model.predict(missing)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(extra)


@pytest.mark.parametrize("multi_class", MULTICLASS)
def test_refit_resets_schema(multi_class, data):
    """Refitting replaces the schema of the previous fit instead of mixing the two."""
    df, y = data

    model = _classifier(DemographicParityClassifier, multi_class, sensitive_cols=0).fit(df, y)
    assert list(model.feature_names_in_) == ["x1", "x2", "x3"]
    assert model.estimators_[0].sensitive_cols_ == ["x1"]

    X_array = df.to_numpy()
    model.fit(X_array, y)
    assert model.n_features_in_ == 3
    assert not hasattr(model, "feature_names_in_")
    assert model.estimators_[0].sensitive_cols_ == [0]
    assert model.coef_names_ == [1, 2]
    assert model.predict(X_array).shape == (len(X_array),)

    # refit on a frame with a different schema
    reduced = df[["x1", "x2"]]
    model.fit(reduced, y)
    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["x1", "x2"]
    assert model.estimators_[0].sensitive_cols_ == ["x1"]
    assert model.coef_names_ == ["x2"]
    assert model.predict(reduced).shape == (len(reduced),)

    # and the old width is now rejected
    with pytest.raises(ValueError):
        model.predict(df)


@pytest.mark.parametrize("make", DATAFRAME_MAKERS.values(), ids=DATAFRAME_MAKERS.keys())
@pytest.mark.parametrize("multi_class", MULTICLASS)
def test_fit_and_predict_do_not_modify_input(multi_class, make, data):
    """Neither fitting nor predicting may change the caller's data."""
    df, y = data
    X = make(df)
    original = X.clone() if make is DATAFRAME_MAKERS["polars"] else X.copy(deep=True)
    y_original = y.copy()

    model = _classifier(DemographicParityClassifier, multi_class, sensitive_cols="x1").fit(X, y)
    model.predict(X)
    model.decision_function(X)
    model.score(X, y)

    if make is DATAFRAME_MAKERS["polars"]:
        assert X.equals(original)
    else:
        pd.testing.assert_frame_equal(X, original)
    np.testing.assert_array_equal(y, y_original)


def test_dropping_the_only_predictor(data):
    """With no predictor left the model is intercept-only, or refuses to fit without an intercept."""
    df, y = data
    X = df[["x1"]]

    without_intercept = _classifier(DemographicParityClassifier, "ovr", sensitive_cols="x1", fit_intercept=False)
    with pytest.raises(ValueError, match="at least 1 feature"):
        without_intercept.fit(X, y)

    # an intercept-only model is valid and reports zero coefficients, consistently with `coef_names_`
    model = _classifier(DemographicParityClassifier, "ovr", sensitive_cols="x1").fit(X, y)
    inner = model.estimators_[0]
    assert inner.coef_.shape == (1, 0)
    assert model.coef_names_ == []
    assert model.predict_proba(X).shape == (len(X), 2)


@pytest.mark.parametrize("make", DATAFRAME_MAKERS.values(), ids=DATAFRAME_MAKERS.keys())
@pytest.mark.parametrize("multi_class", MULTICLASS)
@pytest.mark.parametrize("train_sensitive_cols", [False, True])
def test_decision_function_uses_coef_names_columns(train_sensitive_cols, multi_class, make, data):
    """The score is exactly the linear combination of the columns named in `coef_names_`."""
    df, y = data
    X = make(df)

    model = _classifier(
        DemographicParityClassifier, multi_class, sensitive_cols="x1", train_sensitive_cols=train_sensitive_cols
    ).fit(X, y)
    inner = model.estimators_[0]

    manual = X[inner.coef_names_].to_numpy() @ inner.coef_.ravel() + inner.intercept_[0]
    decision = model.decision_function(X)

    if multi_class == "ovo":
        # `OneVsOneClassifier` reports the vote count plus a monotonic squashing of the binary
        # score, exactly like scikit-learn; its inverse is the binary score.
        binary = model.estimators_[0].decision_function(X)
        confidences = binary.reshape(-1, 1)
        expected = _ovo_binary_decision(confidences)
        np.testing.assert_allclose(decision, expected, atol=1e-9)
    else:
        np.testing.assert_allclose(decision, manual, rtol=1e-9, atol=1e-9)


def _ovo_binary_decision(confidence):
    """Two-class `_ovr_decision_function` result for a single binary sub-estimator."""
    votes = np.zeros((len(confidence), 2))
    votes[confidence[:, 0] > 0, 1] += 1  # the sub-estimator predicts 1 for positive scores
    sum_of_confidences = np.column_stack([-confidence[:, 0], confidence[:, 0]])
    return (votes + sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1)))[:, 1]


def test_pyarrow_dataframe_is_supported(data):
    """PyArrow tables go through the same narwhals path as pandas and Polars."""
    pa = pytest.importorskip("pyarrow")
    df, y = data
    table = pa.Table.from_pandas(df)

    model = _classifier(DemographicParityClassifier, "ovr", sensitive_cols="x1").fit(table, y)
    assert model.coef_names_ == ["x2", "x3"]
    reordered = table.select(["x3", "x1", "x2"])
    np.testing.assert_allclose(model.decision_function(table), model.decision_function(reordered))
    assert model.predict(table[:1]).shape == (1,)

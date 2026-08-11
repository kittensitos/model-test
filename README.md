# model-test

A minimal, end-to-end machine learning project: training and evaluating a
**linear regression** model with scikit-learn.

By default it trains on scikit-learn's bundled **diabetes dataset**
(442 patients, 10 clinical features, target = disease progression one year
after baseline), so it runs out of the box with no data download.

## Project structure

| File | Purpose |
|------|---------|
| `train.py` | Full pipeline: load data → train/test split → fit → evaluate → save model + plot |
| `requirements.txt` | Python dependencies |
| `model.joblib` | Trained model (created by `train.py`) |
| `predicted_vs_actual.png` | Test-set evaluation plot (created by `train.py`) |

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

> On Debian/Ubuntu systems without a virtualenv you may need:
> `pip3 install --user --break-system-packages -r requirements.txt`

## Usage

```bash
python3 train.py
```

This prints the dataset shape, test-set metrics, and model coefficients, then
writes two files: `model.joblib` (the trained model) and
`predicted_vs_actual.png` (a scatter of predictions against true values —
points on the dashed diagonal are perfect predictions).

Example output:

```
Test-set performance:
  R²     0.453   (1.0 = perfect, 0.0 = no better than the mean)
  MAE     42.8   (average absolute error, in target units)
  RMSE    53.9   (like MAE but penalizes large misses more)
```

## How it works

1. **Load** — `load_data()` returns a feature DataFrame `X` and target Series `y`.
2. **Split** — 80% of rows train the model; 20% are held out so metrics measure
   generalization, not memorization (`random_state=42` for reproducibility).
3. **Fit** — `LinearRegression` learns `y = intercept + Σ coef_i · feature_i`.
4. **Evaluate** — R², MAE, and RMSE on the held-out test set.
5. **Inspect** — coefficients are printed largest-effect-first; because the
   model is linear, they are directly interpretable.
6. **Save** — the fitted model goes to `model.joblib`.

## Using your own data

Replace the body of `load_data()` in `train.py`:

```python
def load_data():
    df = pd.read_csv("my_data.csv")
    return df.drop(columns=["target"]), df["target"]
```

Everything downstream (split, fit, metrics, plot, saved model) works unchanged.
Features must be numeric; encode categorical columns first
(ex: `pd.get_dummies`).

## Making predictions with the saved model

```python
import joblib

model = joblib.load("model.joblib")
predictions = model.predict(X_new)  # X_new: same columns as training features
```

## Next steps

If a plain linear fit underperforms on your data, try:

- **Feature scaling + polynomial features** (`PolynomialFeatures` in a `Pipeline`)
  to capture non-linear relationships.
- **Regularized variants** — `Ridge` (L2) or `Lasso` (L1, also does feature
  selection) — drop-in replacements for `LinearRegression`.
- **Cross-validation** (`cross_val_score`) for more robust metrics than a single
  train/test split.

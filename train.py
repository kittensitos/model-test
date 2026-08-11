"""Train and evaluate a linear regression model.

Uses scikit-learn's bundled diabetes dataset (442 patients, 10 clinical
features, target = disease progression one year later). To use your own
data, replace load_data() with something that returns (X, y) — ex: read
a CSV with pandas and split it into feature columns and a target column.

Run:  python3 train.py
Outputs: metrics printed to stdout, model.joblib, predicted_vs_actual.png
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data():
    """Return (X, y) as a DataFrame of features and a Series target.

    Swap this out for your own dataset, e.g.:
        df = pd.read_csv("my_data.csv")
        return df.drop(columns=["target"]), df["target"]
    """
    data = load_diabetes(as_frame=True)
    return data.data, data.target


def main():
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Features: {', '.join(X.columns)}\n")

    # Hold out 20% of the data the model never sees during training,
    # so the metrics measure generalization rather than memorization.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Test-set performance:")
    print(f"  R²    {r2:6.3f}   (1.0 = perfect, 0.0 = no better than the mean)")
    print(f"  MAE   {mae:6.1f}   (average absolute error, in target units)")
    print(f"  RMSE  {rmse:6.1f}   (like MAE but penalizes large misses more)\n")

    # A linear model is y = intercept + sum(coef_i * feature_i), so the
    # coefficients are directly interpretable.
    coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    print(f"Intercept: {model.intercept_:.1f}")
    print("Coefficients (largest effect first):")
    for name, value in coefs.items():
        print(f"  {name:>6}  {value:9.1f}")

    joblib.dump(model, "model.joblib")
    print("\nSaved model to model.joblib")

    plot_predictions(y_test, y_pred, r2)
    print("Saved plot to predicted_vs_actual.png")


def plot_predictions(y_test, y_pred, r2):
    """Scatter of predicted vs. actual values with a y = x reference line."""
    surface = "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=surface)
    ax.set_facecolor(surface)

    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    lims = (lo - pad, hi + pad)

    ax.plot(lims, lims, linestyle="--", linewidth=1, color="#c3c2b7", zorder=1)
    ax.annotate(
        "perfect prediction", xy=(0.97, 0.94), xycoords="axes fraction",
        ha="right", fontsize=9, color="#898781",
    )
    ax.scatter(
        y_test, y_pred, s=50, color="#2a78d6", alpha=0.75,
        edgecolors=surface, linewidths=0.5, zorder=2,
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual disease progression", fontsize=10, color="#52514e")
    ax.set_ylabel("Predicted disease progression", fontsize=10, color="#52514e")
    ax.set_title(
        f"Linear regression on test set (R² = {r2:.2f})",
        fontsize=12, color="#0b0b0b", pad=12,
    )

    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=9)

    fig.tight_layout()
    fig.savefig("predicted_vs_actual.png", dpi=150, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Load processed data
X = pd.read_csv("data/processed_train_X.csv")
y = pd.read_csv("data/processed_train_y.csv").squeeze("columns")
X_test = pd.read_csv("data/processed_test_X.csv")
test_ids = pd.read_csv("data/processed_test_ids.csv").squeeze("columns")

# 2. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define candidate models
models = {
    # Ridge is scale-sensitive (its penalty term treats a "3000 sq ft"
    # column and a "0/1" one-hot column very differently unless scaled first)
    # — wrapping it in a pipeline with StandardScaler fixes the
    # overflow warnings and generally improves its accuracy too.
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    # Tree-based models split on raw thresholds, so scaling doesn't
    # help or hurt them — left as-is.
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}

# 4. Quick single-split check (RMSE on log SalePrice, matches Kaggle metric)
print("=" * 55)
print("SINGLE VALIDATION SPLIT (80/20)")
print("=" * 55)
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"{name:15s} val RMSE (log SalePrice): {rmse:.4f}")

# 5. 5-fold cross-validation for a more reliable estimate
print("\n" + "=" * 55)
print("5-FOLD CROSS-VALIDATION")
print("=" * 55)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
    )
    rmse_scores = np.sqrt(-scores)
    cv_scores[name] = rmse_scores.mean()
    print(f"{name:15s} CV RMSE: {rmse_scores.mean():.4f}  (+/- {rmse_scores.std():.4f})")

# 6. Pick the best model, retrain on full training data
best_name = min(cv_scores, key=cv_scores.get)
best_model = models[best_name]
print(f"\nBest model: {best_name} (CV RMSE: {cv_scores[best_name]:.4f})")

best_model.fit(X, y)

# 7. Predict on test set, reverse log transform, write submission
test_preds_log = best_model.predict(X_test)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("outputs/submission.csv", index=False)

print("\nSaved outputs/submission.csv")
print(submission.head())
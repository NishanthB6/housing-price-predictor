import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)


def load_data(data_dir: str = "data"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


def basic_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("=" * 60)
    print("SHAPES")
    print("=" * 60)
    print(f"train: {train.shape}  (includes SalePrice)")
    print(f"test:  {test.shape}  (no SalePrice — this is what we predict)")

    print("\n" + "=" * 60)
    print("TARGET: SalePrice")
    print("=" * 60)
    print(train["SalePrice"].describe())
    print(f"\nSkew: {train['SalePrice'].skew():.3f}")
    print("(Right-skewed target is common here — consider a log transform,")
    print(" e.g. np.log1p(SalePrice), before modeling.)")

    print("\n" + "=" * 60)
    print("MISSING VALUES (train) — top 20 columns")
    print("=" * 60)
    missing = train.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    missing_pct = (missing / len(train) * 100).round(1)
    print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).head(20))

    print("\nNote: for columns like PoolQC, Alley, Fence, FireplaceQu,")
    print("GarageType, etc., NaN means the feature doesn't exist on the")
    print("property (e.g. no pool) — NOT unknown data. Check")
    print("data_description.txt before imputing these.")

    print("\n" + "=" * 60)
    print("DTYPES SUMMARY")
    print("=" * 60)
    print(train.dtypes.value_counts())

    print("\n" + "=" * 60)
    print("TOP NUMERIC CORRELATIONS WITH SalePrice")
    print("=" * 60)
    numeric = train.select_dtypes(include="number")
    corrs = numeric.corr()["SalePrice"].sort_values(ascending=False)
    print(corrs.head(11))  # includes SalePrice itself at 1.0


if __name__ == "__main__":
    train_df, test_df = load_data()
    basic_overview(train_df, test_df)

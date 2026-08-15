import pandas as pd
import numpy as np

# 1. Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["Id"]

# 2. Drop known outliers
train = train.drop(
    train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index
)

# 3. Separate target, log-transform it, combine train+test for cleaning
y = np.log1p(train["SalePrice"])
train_features = train.drop(columns=["SalePrice", "Id"])
test_features = test.drop(columns=["Id"])

all_data = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# 4. Fill "NA means none" columns
none_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
for col in none_cols:
    all_data[col] = all_data[col].fillna("None")

# 5. Impute missing numeric values
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

zero_cols = [
    "GarageYrBlt", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "GarageArea",
    "BsmtFullBath", "BsmtHalfBath",
]
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

# 6. Encode ordinal quality columns (Po < Fa < TA < Gd < Ex)
qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ordinal_cols = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]
for col in ordinal_cols:
    all_data[col] = all_data[col].fillna("None").map(qual_map)

# 7. Engineer combined features
all_data["TotalSF"] = (
    all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
)
all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
all_data["YearsSinceRemodel"] = all_data["YrSold"] - all_data["YearRemodAdd"]
all_data["TotalBath"] = (
    all_data["FullBath"]
    + 0.5 * all_data["HalfBath"]
    + all_data["BsmtFullBath"]
    + 0.5 * all_data["BsmtHalfBath"]
)

# 8. One-hot encode remaining nominal categorical columns
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)  # code, not magnitude

# Catch any leftover missing values in remaining columns (rare, but a few test-only columns have 1-2 stray NaNs not covered above)
for col in all_data.columns:
    if all_data[col].isnull().any():
        if pd.api.types.is_numeric_dtype(all_data[col]):
            all_data[col] = all_data[col].fillna(all_data[col].median())
        else:
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

all_data = pd.get_dummies(all_data)

# 9. Split back into train/test and save
X = all_data.iloc[: len(train)].reset_index(drop=True)
X_test = all_data.iloc[len(train):].reset_index(drop=True)

X.to_csv("data/processed_train_X.csv", index=False)
y.to_csv("data/processed_train_y.csv", index=False)
X_test.to_csv("data/processed_test_X.csv", index=False)
test_ids.to_csv("data/processed_test_ids.csv", index=False)

print("Done.")
print("X (train features):", X.shape)
print("y (train target):", y.shape)
print("X_test:", X_test.shape)


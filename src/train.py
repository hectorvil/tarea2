import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

PREP_DIR = "data/prep"
ARTIFACTS_DIR = "artifacts"


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(PREP_DIR, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(PREP_DIR, "valid.parquet"))

    with open(os.path.join(PREP_DIR, "meta.json")) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]

    X_train = train_df[feature_cols]
    y_train = train_df["y"].astype(np.float32)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["y"].astype(np.float32)

    cat_features = [c for c in ["shop_id", "item_id", "month", "year"] if c in feature_cols]

    y_train_bin = (y_train > 0).astype(int)
    y_valid_bin = (y_valid > 0).astype(int)

    clf = lgb.LGBMClassifier(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=256,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )

    clf.fit(
        X_train, y_train_bin,
        eval_set=[(X_valid, y_valid_bin)],
        eval_metric="binary_logloss",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    mask_pos_tr = y_train > 0
    mask_pos_va = y_valid > 0

    reg = lgb.LGBMRegressor(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=256,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        objective="regression",
    )

    reg.fit(
        X_train.loc[mask_pos_tr], y_train.loc[mask_pos_tr],
        eval_set=[(X_valid.loc[mask_pos_va], y_valid.loc[mask_pos_va])],
        eval_metric="rmse",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    bundle = {
        "clf": clf,
        "reg": reg,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "meta": meta,
    }

    joblib.dump(bundle, os.path.join(ARTIFACTS_DIR, "model.joblib"))


if __name__ == "__main__":
    main()

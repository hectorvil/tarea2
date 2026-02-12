import os
import json
import gc
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PREP_DIR = "data/prep"


def main() -> None:
    os.makedirs(PREP_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(RAW_DIR, "sales_train.csv"))
    test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    train["date_block_num"] = pd.to_numeric(train["date_block_num"], errors="coerce")
    train["shop_id"] = pd.to_numeric(train["shop_id"], errors="coerce")
    train["item_id"] = pd.to_numeric(train["item_id"], errors="coerce")
    train["item_price"] = pd.to_numeric(train["item_price"], errors="coerce")
    train["item_cnt_day"] = pd.to_numeric(train["item_cnt_day"], errors="coerce")

    test["ID"] = pd.to_numeric(test["ID"], errors="coerce")
    test["shop_id"] = pd.to_numeric(test["shop_id"], errors="coerce")
    test["item_id"] = pd.to_numeric(test["item_id"], errors="coerce")

    train = train.dropna(subset=["date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"])
    test = test.dropna(subset=["ID", "shop_id", "item_id"])

    train["date_block_num"] = train["date_block_num"].astype(np.int16)
    train["shop_id"] = train["shop_id"].astype(np.int16)
    train["item_id"] = train["item_id"].astype(np.int16)

    test["ID"] = test["ID"].astype(np.int32)
    test["shop_id"] = test["shop_id"].astype(np.int16)
    test["item_id"] = test["item_id"].astype(np.int16)

    train = train[train["item_price"] >= 0]
    train = train[train["item_price"] < 100000]
    train = train[(train["item_cnt_day"] > -1000) & (train["item_cnt_day"] < 1000)]

    monthly = (
        train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg(item_cnt_month=("item_cnt_day", "sum"), price_mean=("item_price", "mean"))
    )
    monthly["item_cnt_month"] = monthly["item_cnt_month"].clip(0, 20).astype(np.float32)
    monthly["price_mean"] = monthly["price_mean"].astype(np.float32)

    max_block = int(monthly["date_block_num"].max())
    test_block = max_block + 1

    global_agg = (
        monthly.groupby("date_block_num", as_index=False)
        .agg(global_mean=("item_cnt_month", "mean"), global_sum=("item_cnt_month", "sum"), global_pairs=("item_cnt_month", "size"))
    )
    global_agg["date_block_num"] = (global_agg["date_block_num"] + 1).astype(np.int16)

    item_agg = (
        monthly.groupby(["date_block_num", "item_id"], as_index=False)
        .agg(item_mean=("item_cnt_month", "mean"), item_shops=("shop_id", "nunique"), item_price_mean=("price_mean", "mean"))
    )
    item_agg["date_block_num"] = (item_agg["date_block_num"] + 1).astype(np.int16)

    shop_agg = (
        monthly.groupby(["date_block_num", "shop_id"], as_index=False)
        .agg(shop_mean=("item_cnt_month", "mean"), shop_items=("item_id", "nunique"))
    )
    shop_agg["date_block_num"] = (shop_agg["date_block_num"] + 1).astype(np.int16)

    test_pairs = test[["shop_id", "item_id"]].drop_duplicates()
    n_pairs = len(test_pairs)
    months = np.arange(0, test_block + 1, dtype=np.int16)
    n_months = len(months)

    panel = pd.DataFrame(
        {
            "date_block_num": np.tile(months, n_pairs).astype(np.int16),
            "shop_id": np.repeat(test_pairs["shop_id"].values, n_months).astype(np.int16),
            "item_id": np.repeat(test_pairs["item_id"].values, n_months).astype(np.int16),
        }
    )

    panel = panel.merge(
        monthly[["date_block_num", "shop_id", "item_id", "item_cnt_month", "price_mean"]],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

    panel["item_cnt_month"] = panel["item_cnt_month"].fillna(0).astype(np.float32)
    panel["price_mean"] = panel["price_mean"].astype(np.float32)

    panel["month"] = (panel["date_block_num"] % 12).astype(np.int8)
    panel["year"] = (panel["date_block_num"] // 12).astype(np.int8)

    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12).astype(np.float32)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12).astype(np.float32)

    panel = panel.merge(global_agg, on="date_block_num", how="left")
    panel = panel.merge(item_agg, on=["date_block_num", "item_id"], how="left")
    panel = panel.merge(shop_agg, on=["date_block_num", "shop_id"], how="left")

    for c in ["global_mean", "global_sum", "global_pairs", "item_mean", "item_shops", "item_price_mean", "shop_mean", "shop_items"]:
        panel[c] = panel[c].fillna(0).astype(np.float32)

    del train, monthly, global_agg, item_agg, shop_agg, test_pairs
    gc.collect()

    p_obs = panel["price_mean"].dropna()
    p999 = p_obs.quantile(0.999) if len(p_obs) else 0.0

    price_clip = panel["price_mean"].clip(0, p999)
    panel["log_price_obs"] = np.where(panel["price_mean"].notna(), np.log1p(price_clip), np.nan).astype(np.float32)
    panel["item_log_price_mean"] = np.log1p(panel["item_price_mean"].clip(0, p999)).astype(np.float32)

    panel.sort_values(["shop_id", "item_id", "date_block_num"], inplace=True, ignore_index=True)
    g = panel.groupby(["shop_id", "item_id"], sort=False)

    panel["log_price_last"] = g["log_price_obs"].ffill()
    panel["log_price_last"] = g["log_price_last"].shift(1).astype(np.float32)
    panel["price_missing_last"] = panel["log_price_last"].isna().astype(np.int8)
    panel["log_price_last"] = panel["log_price_last"].fillna(panel["item_log_price_mean"]).astype(np.float32)
    panel["price_gap_item"] = (panel["log_price_last"] - panel["item_log_price_mean"]).astype(np.float32)

    for lag in [1, 2, 3, 4, 5, 6, 12]:
        panel[f"cnt_lag_{lag}"] = g["item_cnt_month"].shift(lag).fillna(0).astype(np.float32)

    eps = 1e-6
    l1 = panel["cnt_lag_1"]; l2 = panel["cnt_lag_2"]; l3 = panel["cnt_lag_3"]
    l4 = panel["cnt_lag_4"]; l5 = panel["cnt_lag_5"]; l6 = panel["cnt_lag_6"]
    l12 = panel["cnt_lag_12"]

    panel["sum_3"] = (l1 + l2 + l3).astype(np.float32)
    panel["mean_3"] = (panel["sum_3"] / 3.0).astype(np.float32)
    panel["max_3"] = np.maximum.reduce([l1.values, l2.values, l3.values]).astype(np.float32)
    panel["nz_3"] = ((l1 > 0).astype(np.int8) + (l2 > 0).astype(np.int8) + (l3 > 0).astype(np.int8)).astype(np.int8)
    mean_sq_3 = ((l1 * l1 + l2 * l2 + l3 * l3) / 3.0).astype(np.float32)
    panel["std_3"] = np.sqrt(np.maximum(mean_sq_3 - panel["mean_3"] * panel["mean_3"], 0)).astype(np.float32)

    panel["sum_6"] = (l1 + l2 + l3 + l4 + l5 + l6).astype(np.float32)
    panel["mean_6"] = (panel["sum_6"] / 6.0).astype(np.float32)
    panel["max_6"] = np.maximum.reduce([l1.values, l2.values, l3.values, l4.values, l5.values, l6.values]).astype(np.float32)
    panel["nz_6"] = ((l1 > 0) + (l2 > 0) + (l3 > 0) + (l4 > 0) + (l5 > 0) + (l6 > 0)).astype(np.int8)
    mean_sq_6 = ((l1 * l1 + l2 * l2 + l3 * l3 + l4 * l4 + l5 * l5 + l6 * l6) / 6.0).astype(np.float32)
    panel["std_6"] = np.sqrt(np.maximum(mean_sq_6 - panel["mean_6"] * panel["mean_6"], 0)).astype(np.float32)

    panel["rate_6"] = (panel["nz_6"] / 6.0).astype(np.float32)
    panel["mean_nonzero_6"] = (panel["sum_6"] / (panel["nz_6"].astype(np.float32) + eps)).astype(np.float32)
    panel["interval_6"] = (6.0 / (panel["nz_6"].astype(np.float32) + eps)).astype(np.float32)

    panel["active_1"] = (l1 > 0).astype(np.int8)
    panel["dead_6"] = (panel["nz_6"] == 0).astype(np.int8)

    panel["trend_1_3"] = (l1 - l3).astype(np.float32)
    panel["trend_1_12"] = (l1 - l12).astype(np.float32)
    panel["ratio_1_12"] = (l1 / (l12 + eps)).astype(np.float32)

    sale_month = panel["date_block_num"].where(panel["item_cnt_month"] > 0, np.nan)
    last_sale_inclusive = sale_month.groupby([panel["shop_id"], panel["item_id"]]).ffill()
    panel["last_sale_month"] = last_sale_inclusive.groupby([panel["shop_id"], panel["item_id"]]).shift(1)
    panel["recency"] = (panel["date_block_num"] - panel["last_sale_month"]).fillna(99).clip(0, 99).astype(np.int16)
    panel.drop(columns=["last_sale_month"], inplace=True)

    panel["sold"] = (panel["item_cnt_month"] > 0).astype(np.int8)
    panel["sold_cum"] = g["sold"].cumsum().astype(np.int16)
    panel["sold_cum_lag1"] = g["sold_cum"].shift(1).fillna(0).astype(np.int16)

    panel["sales_cum"] = g["item_cnt_month"].cumsum().astype(np.float32)
    panel["sales_cum_lag1"] = g["sales_cum"].shift(1).fillna(0).astype(np.float32)
    panel["log_sales_cum_lag1"] = np.log1p(panel["sales_cum_lag1"]).astype(np.float32)

    panel["never_sold_before"] = (panel["sold_cum_lag1"] == 0).astype(np.int8)

    train_df = panel[panel["date_block_num"] <= max_block - 1].copy()
    valid_df = panel[panel["date_block_num"] == max_block].copy()
    test_df = panel[panel["date_block_num"] == test_block].copy()

    y_train = train_df["item_cnt_month"].astype(np.float32)
    y_valid = valid_df["item_cnt_month"].astype(np.float32)

    feature_cols = [
        "date_block_num", "month", "year", "month_sin", "month_cos", "shop_id", "item_id",
        "global_mean", "global_sum", "global_pairs",
        "item_mean", "item_shops",
        "shop_mean", "shop_items",
        "log_price_last", "item_log_price_mean", "price_gap_item", "price_missing_last",
        "cnt_lag_1", "cnt_lag_2", "cnt_lag_3", "cnt_lag_4", "cnt_lag_5", "cnt_lag_6", "cnt_lag_12",
        "sum_3", "mean_3", "std_3", "max_3", "nz_3",
        "sum_6", "mean_6", "std_6", "max_6", "nz_6",
        "rate_6", "mean_nonzero_6", "interval_6",
        "active_1", "dead_6",
        "recency", "trend_1_3", "trend_1_12", "ratio_1_12",
        "sold_cum_lag1", "log_sales_cum_lag1", "never_sold_before",
    ]

    X_train = train_df[feature_cols]
    X_valid = valid_df[feature_cols]
    X_test = test_df[feature_cols]

    train_out = X_train.copy()
    train_out["y"] = y_train.values

    valid_out = X_valid.copy()
    valid_out["y"] = y_valid.values

    test_out = X_test.copy()
    test_meta = test_df[["shop_id", "item_id"]].copy()

    train_out.to_parquet(os.path.join(PREP_DIR, "train.parquet"), index=False)
    valid_out.to_parquet(os.path.join(PREP_DIR, "valid.parquet"), index=False)
    test_out.to_parquet(os.path.join(PREP_DIR, "test_features.parquet"), index=False)
    test_meta.to_parquet(os.path.join(PREP_DIR, "test_pairs.parquet"), index=False)

    meta = {"feature_cols": feature_cols, "max_block": int(max_block), "test_block": int(test_block)}
    with open(os.path.join(PREP_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

"""Gradient-boosted price model on HM Land Registry Price Paid Data (Greater London, 2023 onwards).
Temporal train/test split, a simple baseline, SHAP explanations and error by borough.
Run after clean.py. Writes models/, charts/06-10*.png and data/processed/model_*.{csv,json}."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, pathlib, warnings
import numpy as np, pandas as pd, lightgbm as lgb, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
INT, PROC, CH, MOD = ROOT / "data" / "interim", ROOT / "data" / "processed", ROOT / "charts", ROOT / "models"
MOD.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "font.size": 10})
NAVY, RED, TEAL, GREY = "#1f3b73", "#c8102e", "#00857c", "#7f7f7f"
CAT = ["property_type", "tenure", "borough", "postcode_district"]
FEATURES = CAT + ["new_build", "months_since_start"]
START = pd.Timestamp("2023-01-01")
TEST_FROM = pd.Timestamp("2025-07-01")    # train: Jan 2023 - Jun 2025, test: Jul 2025 onwards
VALID_FROM = pd.Timestamp("2025-04-01")   # last 3 training months used for early stopping
MIN_PRICE = 25_000                        # drop nominal / non-market prices

df = pd.read_csv(INT / "ppd_london.csv", parse_dates=["date"])
n_all = len(df)
df = df[df.price >= MIN_PRICE].copy()
df["months_since_start"] = (df.date.dt.year - START.year) * 12 + df.date.dt.month - START.month + (df.date.dt.day - 1) / 31
for c in CAT:
    df[c] = df[c].astype("category")
df["log_price"] = np.log(df.price)

train, test = df[df.date < TEST_FROM], df[df.date >= TEST_FROM]
fit, valid = train[train.date < VALID_FROM], train[train.date >= VALID_FROM]
params = dict(objective="regression_l1", learning_rate=0.05, num_leaves=127, min_child_samples=30,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, cat_smooth=10, verbose=-1, seed=42)
dtrain = lgb.Dataset(fit[FEATURES], fit.log_price, categorical_feature=CAT)
dvalid = lgb.Dataset(valid[FEATURES], valid.log_price, reference=dtrain)
booster = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
best_iter = booster.best_iteration
# Refit on the full training period (Jan 2023 - Jun 2025) with the chosen number of rounds
booster = lgb.train(params, lgb.Dataset(train[FEATURES], train.log_price, categorical_feature=CAT), num_boost_round=best_iter)
booster.save_model(str(MOD / "lgbm_price_model.txt"))

test = test.copy()
test["pred"] = np.exp(booster.predict(test[FEATURES]))
# Baseline: training-period median price for the same borough and property type
base = train.groupby(["borough", "property_type"], observed=True).price.median().rename("baseline").reset_index()
test = test.merge(base, on=["borough", "property_type"], how="left")
test["baseline"] = test.baseline.fillna(train.price.median())

def metrics(y, p):
    ape = (p - y).abs() / y
    ly, lp = np.log(y), np.log(p)
    return {"mae_gbp": int((p - y).abs().mean()), "median_abs_error_gbp": int((p - y).abs().median()),
            "median_ape_pct": round(100 * ape.median(), 1), "mape_pct": round(100 * ape.mean(), 1),
            "within_10pct": round(100 * (ape <= .10).mean(), 1), "within_20pct": round(100 * (ape <= .20).mean(), 1),
            "r2_log": round(1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum(), 3)}
M = {"model": metrics(test.price, test.pred), "baseline_borough_type_median": metrics(test.price, test.baseline)}

# Error by borough
test["ape"] = (test.pred - test.price).abs() / test.price
test["ape_base"] = (test.baseline - test.price).abs() / test.price
eb = test.groupby("borough", observed=True).agg(sales=("price", "size"), median_price=("price", "median"),
                                                 median_ape_pct=("ape", "median"), baseline_median_ape_pct=("ape_base", "median"),
                                                 bias_pct=("pred", lambda s: 0)).drop(columns="bias_pct")
eb["median_ape_pct"] = (100 * eb.median_ape_pct).round(1); eb["baseline_median_ape_pct"] = (100 * eb.baseline_median_ape_pct).round(1)
eb["median_signed_error_pct"] = (100 * test.assign(e=(test.pred - test.price) / test.price).groupby("borough", observed=True).e.median()).round(1)
eb = eb.sort_values("median_ape_pct")
eb.to_csv(PROC / "model_error_by_borough.csv")
et = test.groupby("property_type", observed=True).agg(sales=("price", "size"), median_ape_pct=("ape", "median"))
et["median_ape_pct"] = (100 * et.median_ape_pct).round(1)
et.to_csv(PROC / "model_error_by_type.csv")

# Feature importance (gain) and SHAP
imp = pd.Series(booster.feature_importance("gain"), index=FEATURES)
imp = (100 * imp / imp.sum()).round(1).sort_values()
sample = test.sample(4000, random_state=42)
sv = shap.TreeExplainer(booster).shap_values(sample[FEATURES])
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values()
pd.DataFrame({"gain_share_pct": imp, "mean_abs_shap_log": mean_abs.round(4)}).to_csv(PROC / "model_feature_importance.csv")
# SHAP effect of each category for the categorical features (mean SHAP value -> % price effect)
eff = {}
for c in ["property_type", "tenure"]:
    j = FEATURES.index(c)
    e = pd.Series(sv[:, j], index=sample[c].astype(str).values).groupby(level=0).mean()
    eff[c] = {k: round((np.exp(v) - 1) * 100, 1) for k, v in e.items()}
jn = FEATURES.index("new_build")
nbe = pd.Series(sv[:, jn], index=sample.new_build.values).groupby(level=0).mean()
eff["new_build_vs_resale_pct"] = round((np.exp(nbe[1] - nbe[0]) - 1) * 100, 1)
jb = FEATURES.index("borough")
be = pd.Series(sv[:, jb], index=sample.borough.astype(str).values).groupby(level=0).mean().sort_values()
be_pct = ((np.exp(be) - 1) * 100).round(1)
be_pct.rename("shap_effect_pct").to_csv(PROC / "model_borough_shap_effect.csv")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].barh(imp.index, imp.values, color=NAVY); ax[0].set(title="Feature importance (share of total gain, %)")
ax[1].barh(mean_abs.index, mean_abs.values, color=TEAL); ax[1].set(title="Mean |SHAP value| (log price) on 4,000 test sales")
fig.tight_layout(); fig.savefig(CH / "06_feature_importance_shap.png"); plt.close(fig)

plt.figure()
shap.summary_plot(sv, sample[FEATURES].assign(**{c: sample[c].cat.codes for c in CAT}), show=False, max_display=6, plot_size=(9, 4.5))
plt.title("SHAP summary (categorical features shown by category code)", fontsize=10)
plt.tight_layout(); plt.savefig(CH / "07_shap_summary.png"); plt.close("all")

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(eb.index.astype(str), eb.median_ape_pct, color=NAVY, label="Gradient boosting")
ax.scatter(eb.baseline_median_ape_pct, eb.index.astype(str), color=RED, zorder=3, s=18, label="Baseline (borough × type median)")
ax.xaxis.set_major_formatter(mt.PercentFormatter()); ax.legend(loc="lower right", fontsize=8)
ax.set(title=f"Median absolute % error by borough, test set ({TEST_FROM:%b %Y} onwards)")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(CH / "08_error_by_borough.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 6))
s = test.sample(8000, random_state=1)
ax.scatter(s.price, s.pred, s=3, alpha=.25, color=NAVY)
lim = [5e4, 1e7]; ax.plot(lim, lim, color=RED, lw=1)
ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Actual price", ylabel="Predicted price",
       title="Predicted vs actual, test set (8,000 sampled sales)")
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"£{x/1e3:,.0f}k")); ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"£{x/1e3:,.0f}k"))
fig.tight_layout(); fig.savefig(CH / "09_pred_vs_actual.png"); plt.close(fig)

be2 = be_pct.sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(be2.index, be2.values, color=[RED if v > 0 else TEAL for v in be2.values])
ax.xaxis.set_major_formatter(mt.PercentFormatter())
ax.set(title="Borough effect on price, holding other features fixed (mean SHAP, % vs average)")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(CH / "10_borough_shap_effect.png"); plt.close(fig)

# Metadata for the Streamlit estimator
pc = df.groupby(["borough", "postcode_district"], observed=True).size().rename("n").reset_index()
pc = pc[pc.n >= 20]
meta = {"features": FEATURES, "categorical": CAT, "start": str(START.date()), "test_from": str(TEST_FROM.date()),
        "train_rows": int(len(train)), "test_rows": int(len(test)), "rows_dropped_below_min_price": int(n_all - len(df)),
        "min_price": MIN_PRICE, "best_iteration": int(best_iter), "params": params,
        "train_period": f"{train.date.min():%d %b %Y} to {train.date.max():%d %b %Y}",
        "test_period": f"{test.date.min():%d %b %Y} to {test.date.max():%d %b %Y}",
        "latest_months_since_start": round(float(df.months_since_start.max()), 2),
        "categories": {c: [str(x) for x in df[c].cat.categories] for c in CAT},
        "borough_postcodes": {b: sorted(g.postcode_district.astype(str)) for b, g in pc.groupby("borough", observed=True)},
        "metrics": M, "effects": eff,
        "best_borough_mape": {"borough": str(eb.index[0]), "median_ape_pct": float(eb.median_ape_pct.iloc[0])},
        "worst_borough_mape": {"borough": str(eb.index[-1]), "median_ape_pct": float(eb.median_ape_pct.iloc[-1])},
        "boroughs_model_beats_baseline": int((eb.median_ape_pct < eb.baseline_median_ape_pct).sum()),
        "importance_gain_pct": imp.sort_values(ascending=False).to_dict(),
        "borough_effect_top": be_pct.sort_values(ascending=False).head(3).to_dict(),
        "borough_effect_bottom": be_pct.sort_values().head(3).to_dict()}
(MOD / "model_meta.json").write_text(json.dumps(meta, indent=2))
(PROC / "model_metrics.json").write_text(json.dumps({k: meta[k] for k in meta if k not in ("categories", "borough_postcodes", "params")}, indent=2))
print(json.dumps({k: meta[k] for k in meta if k not in ("categories", "borough_postcodes")}, indent=2))
print(eb.to_string()); print(et.to_string())

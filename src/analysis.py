"""Borough-level price trends (UK HPI), affordability (ONS ratios) and transaction summaries (Price Paid Data).
Run after clean.py. Writes charts/0[1-5]*.png, data/processed summary CSVs and data/processed/findings_trends.json."""
import json, pathlib, warnings
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
INT, PROC, CH = ROOT / "data" / "interim", ROOT / "data" / "processed", ROOT / "charts"
CH.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "font.size": 10})
NAVY, RED, TEAL, GOLD, GREY = "#1f3b73", "#c8102e", "#00857c", "#e3a21a", "#7f7f7f"
GBP = mt.FuncFormatter(lambda x, _: f"£{x/1e3:,.0f}k" if x < 1e6 else f"£{x/1e6:,.1f}m")

hpi = pd.read_csv(PROC / "hpi_london.csv", parse_dates=["month"])
aff = pd.read_csv(PROC / "affordability.csv")
ppd = pd.read_csv(INT / "ppd_london.csv", parse_dates=["date"])
F = {}

# 1. London vs England, long run -----------------------------------------------------
lon = hpi[hpi.area_code == "E12000007"].set_index("month"); eng = hpi[hpi.area_code == "E92000001"].set_index("month")
last = hpi.month.max()
lon_peak_m = lon.average_price.idxmax()
F.update(hpi_latest_month=last.strftime("%B %Y"), london_avg_latest=int(lon.loc[last, "average_price"]),
         london_annual_change=float(lon.loc[last, "annual_change_pct"]),
         england_avg_latest=int(eng.loc[last, "average_price"]), england_annual_change=float(eng.loc[last, "annual_change_pct"]),
         london_peak=int(lon.average_price.max()), london_peak_month=lon_peak_m.strftime("%B %Y"),
         london_vs_peak_pct=round((lon.loc[last, "average_price"] / lon.average_price.max() - 1) * 100, 1),
         london_multiple_since_jan1995=round(lon.loc[last, "average_price"] / lon.loc["1995-01-01", "average_price"], 2),
         london_10y_change_pct=round((lon.loc[last, "average_price"] / lon.loc[last - pd.DateOffset(years=10), "average_price"] - 1) * 100, 1),
         england_10y_change_pct=round((eng.loc[last, "average_price"] / eng.loc[last - pd.DateOffset(years=10), "average_price"] - 1) * 100, 1),
         london_premium_vs_england_pct=round((lon.loc[last, "average_price"] / eng.loc[last, "average_price"] - 1) * 100, 1))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(lon.index, lon.average_price, color=RED, lw=2, label="London")
ax.plot(eng.index, eng.average_price, color=NAVY, lw=2, label="England")
ax.yaxis.set_major_formatter(GBP); ax.legend()
ax.set(title=f"Average house price, Jan 1995 to {last:%b %Y} (UK House Price Index)")
fig.tight_layout(); fig.savefig(CH / "01_london_vs_england.png"); plt.close(fig)

# 2. Borough league table -----------------------------------------------------------
b = hpi[hpi.area_code.str.startswith("E09")]
cur = b[b.month == last].set_index("area")
ten = b[b.month == last - pd.DateOffset(years=10)].set_index("area")
pk = b.groupby("area").apply(lambda g: g.loc[g.average_price.idxmax()])
tbl = pd.DataFrame({"area_code": cur.area_code, "average_price": cur.average_price, "annual_change_pct": cur.annual_change_pct,
                    "change_10y_pct": ((cur.average_price / ten.average_price - 1) * 100).round(1),
                    "peak_price": pk.average_price, "peak_month": pk.month.dt.strftime("%Y-%m"),
                    "vs_peak_pct": ((cur.average_price / pk.average_price - 1) * 100).round(1),
                    "flat": cur.flat, "terraced": cur.terraced, "semi_detached": cur.semi_detached, "detached": cur.detached})
tbl = tbl.sort_values("average_price", ascending=False)
tbl.reset_index().to_csv(PROC / "borough_summary.csv", index=False)
F.update(most_expensive=tbl.index[0], most_expensive_price=int(tbl.average_price.iloc[0]),
         cheapest=tbl.index[-1], cheapest_price=int(tbl.average_price.iloc[-1]),
         price_gap_multiple=round(tbl.average_price.iloc[0] / tbl.average_price.iloc[-1], 1),
         boroughs_falling_yoy=int((tbl.annual_change_pct < 0).sum()),
         best_10y=tbl.change_10y_pct.idxmax(), best_10y_pct=float(tbl.change_10y_pct.max()),
         worst_10y=tbl.change_10y_pct.idxmin(), worst_10y_pct=float(tbl.change_10y_pct.min()),
         boroughs_below_peak_10pct=int((tbl.vs_peak_pct <= -10).sum()))
t = tbl.sort_values("average_price")
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(t.index, t.average_price, color=[RED if v < 0 else NAVY for v in t.annual_change_pct])
for y, (v, c) in enumerate(zip(t.average_price, t.annual_change_pct)):
    ax.text(v * 1.01, y, f"{c:+.1f}%", va="center", fontsize=7)
ax.xaxis.set_major_formatter(GBP)
ax.set(title=f"Average price by borough, {last:%b %Y} (label = annual change; red = falling)")
fig.tight_layout(); fig.savefig(CH / "02_borough_prices.png"); plt.close(fig)

t = tbl.sort_values("change_10y_pct")
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(t.index, t.change_10y_pct, color=TEAL)
ax.axvline(F["london_10y_change_pct"], color=RED, ls="--", lw=1, label=f"London {F['london_10y_change_pct']:+.1f}%")
ax.axvline(F["england_10y_change_pct"], color=NAVY, ls=":", lw=1, label=f"England {F['england_10y_change_pct']:+.1f}%")
ax.xaxis.set_major_formatter(mt.PercentFormatter()); ax.legend(loc="lower right")
ax.set(title=f"Change in average price over 10 years ({last - pd.DateOffset(years=10):%b %Y} to {last:%b %Y})")
fig.tight_layout(); fig.savefig(CH / "03_borough_10y_change.png"); plt.close(fig)

# 3. Affordability (ONS) ------------------------------------------------------------
a = aff.dropna(subset=["ratio"])
ay = int(a.year.max())
al = a[(a.year == ay) & a.area_code.str.startswith("E09")].sort_values("ratio")
lon_a = a[a.area == "London"].set_index("year").ratio; eng_a = a[a.area == "England"].set_index("year").ratio
F.update(afford_year=ay, london_ratio=float(lon_a[ay]), england_ratio=float(eng_a[ay]), london_ratio_1997=float(lon_a[1997]),
         london_ratio_peak=float(lon_a.max()), london_ratio_peak_year=int(lon_a.idxmax()),
         least_affordable=al.area.iloc[-1], least_affordable_ratio=float(al.ratio.iloc[-1]),
         most_affordable=al.area.iloc[0], most_affordable_ratio=float(al.ratio.iloc[0]),
         boroughs_ratio_over_10=int((al.ratio > 10).sum()), boroughs_total=int(len(al)))
fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].barh(al.area, al.ratio, color=NAVY)
ax[0].axvline(eng_a[ay], color=GREY, ls=":", label=f"England {eng_a[ay]:.1f}")
ax[0].axvline(lon_a[ay], color=RED, ls="--", label=f"London {lon_a[ay]:.1f}")
ax[0].legend(loc="lower right"); ax[0].set(title=f"Median house price ÷ median workplace earnings, {ay}")
ax[0].tick_params(axis="y", labelsize=7)
ax[1].plot(lon_a.index, lon_a.values, color=RED, lw=2, label="London")
ax[1].plot(eng_a.index, eng_a.values, color=NAVY, lw=2, label="England")
ax[1].legend(); ax[1].set(title="Affordability ratio over time (higher = less affordable)")
fig.tight_layout(); fig.savefig(CH / "04_affordability.png"); plt.close(fig)

# 4. Transactions (Price Paid Data) -------------------------------------------------
ppd["year"] = ppd.date.dt.year
by_type = ppd[ppd.year == 2025].groupby("property_type").price.agg(["median", "count"]).round(0)
by_type.to_csv(PROC / "ppd_2025_by_type.csv")
yearly = ppd.groupby("year").price.agg(transactions="count", median_price="median").reset_index()
yearly.to_csv(PROC / "ppd_yearly.csv", index=False)
nb = ppd[ppd.year.isin([2024, 2025])].groupby(["property_type", "new_build"]).price.median().unstack()
F.update(ppd_rows=int(len(ppd)), ppd_first=ppd.date.min().strftime("%d %B %Y"), ppd_last=ppd.date.max().strftime("%d %B %Y"),
         ppd_2025_sales=int((ppd.year == 2025).sum()), ppd_2025_median=int(ppd[ppd.year == 2025].price.median()),
         ppd_2025_flat_share_pct=round(100 * (ppd[ppd.year == 2025].property_type == "Flat/maisonette").mean(), 1),
         ppd_2025_median_by_type={k: int(v) for k, v in by_type["median"].items()},
         flat_newbuild_premium_pct=round((nb.loc["Flat/maisonette", 1] / nb.loc["Flat/maisonette", 0] - 1) * 100, 1))
mb = ppd[ppd.year == 2025].groupby(["borough", "property_type"]).price.median().unstack()
mb.round(0).to_csv(PROC / "ppd_2025_borough_type_median.csv")
fig, ax = plt.subplots(figsize=(10, 4.5))
m = ppd.set_index("date").groupby("property_type").price.resample("QS").median().unstack(0)
m = m[m.index < ppd.date.max().to_period("Q").start_time]  # drop the incomplete latest quarter
for c, col in zip([NAVY, TEAL, GOLD, RED], ["Detached", "Semi-detached", "Terraced", "Flat/maisonette"]):
    ax.plot(m.index, m[col], marker="o", color=c, label=col)
ax.yaxis.set_major_formatter(GBP); ax.legend(fontsize=8)
ax.set(title="Median London sale price by property type, by quarter (Price Paid Data)")
fig.tight_layout(); fig.savefig(CH / "05_ppd_median_by_type.png"); plt.close(fig)

(PROC / "findings_trends.json").write_text(json.dumps(F, indent=2, default=str))
print(json.dumps(F, indent=2, default=str)); print(tbl.head(5).to_string()); print(tbl.tail(5).to_string())

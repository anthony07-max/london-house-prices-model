"""Streamlit app: London house prices explorer and price estimator. Run: streamlit run app/streamlit_app.py"""
import json, pathlib
import numpy as np, pandas as pd, lightgbm as lgb, plotly.express as px, streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
P, MOD = ROOT / "data" / "processed", ROOT / "models"
st.set_page_config(page_title="London House Prices", layout="wide")

@st.cache_data
def load():
    return (pd.read_csv(P / "hpi_london.csv", parse_dates=["month"]), pd.read_csv(P / "borough_summary.csv"),
            pd.read_csv(P / "affordability.csv"), pd.read_csv(P / "model_error_by_borough.csv"),
            json.loads((P / "findings_trends.json").read_text()), json.loads((MOD / "model_meta.json").read_text()))

@st.cache_resource
def model():
    return lgb.Booster(model_file=str(MOD / "lgbm_price_model.txt"))

hpi, bs, aff, err, F, meta = load()
booster = model()
st.title("London house prices: trends, affordability and a price estimator")
st.caption("HM Land Registry Price Paid Data and UK House Price Index; ONS affordability ratios. Contains HM Land Registry data "
           "© Crown copyright and database right 2026, licensed under the Open Government Licence v3.0. "
           "Built by Anthony Adusei-Poku (github.com/anthony07-max).")
c = st.columns(4)
c[0].metric(f"London average ({F['hpi_latest_month']})", f"£{F['london_avg_latest']:,}", f"{F['london_annual_change']:+.1f}% year on year")
c[1].metric("vs London peak", f"{F['london_vs_peak_pct']:+.1f}%", f"peak {F['london_peak_month']}", delta_color="off")
c[2].metric(f"Affordability ratio ({F['afford_year']})", f"{F['london_ratio']}x", f"England {F['england_ratio']}x", delta_color="off")
c[3].metric("Model median error (test)", f"{meta['metrics']['model']['median_ape_pct']}%",
            f"baseline {meta['metrics']['baseline_borough_type_median']['median_ape_pct']}%", delta_color="off")

t1, t2, t3, t4 = st.tabs(["Price estimator", "Borough trends", "Affordability", "Model performance"])
with t1:
    st.subheader("Estimate a sale price")
    a, b = st.columns(2)
    boroughs = sorted(meta["borough_postcodes"])
    borough = a.selectbox("Borough", boroughs, index=boroughs.index("Hackney") if "Hackney" in boroughs else 0)
    pcd = a.selectbox("Postcode district", meta["borough_postcodes"][borough])
    ptype = b.selectbox("Property type", ["Flat/maisonette", "Terraced", "Semi-detached", "Detached"])
    tenure = b.radio("Tenure", ["Leasehold", "Freehold"] if ptype == "Flat/maisonette" else ["Freehold", "Leasehold"], horizontal=True)
    new = b.checkbox("New build")
    x = pd.DataFrame([{"property_type": ptype, "tenure": tenure, "borough": borough, "postcode_district": pcd,
                       "new_build": int(new), "months_since_start": meta["latest_months_since_start"]}])
    for col in meta["categorical"]:
        x[col] = pd.Categorical(x[col], categories=meta["categories"][col])
    est = float(np.exp(booster.predict(x[meta["features"]])[0]))
    e = err.set_index("borough").loc[borough]
    lo, hi = est * (1 - e.median_ape_pct / 100), est * (1 + e.median_ape_pct / 100)
    st.metric("Estimated price", f"£{est:,.0f}")
    st.write(f"Typical error in {borough} on unseen sales is ±{e.median_ape_pct}% (median absolute % error), "
             f"so half of similar sales would land roughly between **£{lo:,.0f}** and **£{hi:,.0f}**.")
    st.info("The model only knows location, type, tenure, new-build status and sale date. It does not know floor area, "
            "bedrooms or condition, so treat this as a starting point for the area rather than a valuation.")
with t2:
    sel = st.multiselect("Boroughs", sorted(bs.area), default=["Hackney", "Newham", "Wandsworth", "Barking and Dagenham"])
    yr = st.slider("From year", 1995, 2026, 2010)
    d = hpi[(hpi.area.isin(sel + ["London"])) & (hpi.month.dt.year >= yr)]
    st.plotly_chart(px.line(d, x="month", y="average_price", color="area", labels={"average_price": "Average price (£)"}), width="stretch")
    st.dataframe(bs, width="stretch", hide_index=True)
with t3:
    yy = st.slider("Year", int(aff.year.min()), int(aff.year.max()), int(aff.year.max()))
    d = aff[(aff.year == yy) & aff.area_code.str.startswith("E09")].sort_values("ratio")
    st.plotly_chart(px.bar(d, x="ratio", y="area", orientation="h", height=750,
                           title=f"Median house price ÷ median workplace-based earnings, {yy}"), width="stretch")
    st.plotly_chart(px.line(aff[aff.area.isin(["London", "England"])], x="year", y="ratio", color="area"), width="stretch")
with t4:
    m = pd.DataFrame(meta["metrics"]).T
    st.write(f"Trained on {meta['train_period']} ({meta['train_rows']:,} sales), tested on {meta['test_period']} ({meta['test_rows']:,} sales).")
    st.dataframe(m, width="stretch")
    st.plotly_chart(px.bar(err.sort_values("median_ape_pct"), x="median_ape_pct", y="borough", orientation="h", height=750,
                           title="Median absolute % error by borough (test set)"), width="stretch")
    st.image(str(ROOT / "charts" / "06_feature_importance_shap.png"))

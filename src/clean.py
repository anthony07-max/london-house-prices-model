"""Tidy the raw files into analysis-ready tables.
- data/interim/ppd_london.csv      transaction-level modelling table (not committed: large, address-adjacent)
- data/processed/hpi_london.csv    monthly UK HPI for the 33 London boroughs, London and England
- data/processed/affordability.csv ONS median price to median workplace earnings ratios, London boroughs
"""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "data" / "processed"
INT.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)
TYPES = {"D": "Detached", "S": "Semi-detached", "T": "Terraced", "F": "Flat/maisonette"}

def clean_hpi():
    cols = ["Date", "RegionName", "AreaCode", "AveragePrice", "Index", "12m%Change", "SalesVolume",
            "DetachedPrice", "SemiDetachedPrice", "TerracedPrice", "FlatPrice"]
    h = pd.read_csv(RAW / "uk_hpi_full.csv", usecols=cols)
    h = h[h.AreaCode.str.startswith("E09") | h.AreaCode.isin(["E12000007", "E92000001"])].copy()
    h["month"] = pd.to_datetime(h.Date, format="%d/%m/%Y")
    h = h.rename(columns={"RegionName": "area", "AreaCode": "area_code", "AveragePrice": "average_price",
                          "Index": "index", "12m%Change": "annual_change_pct", "SalesVolume": "sales_volume",
                          "DetachedPrice": "detached", "SemiDetachedPrice": "semi_detached",
                          "TerracedPrice": "terraced", "FlatPrice": "flat"}).drop(columns="Date")
    h = h[h.month >= "1995-01-01"].sort_values(["area_code", "month"])
    h.to_csv(PROC / "hpi_london.csv", index=False)
    return h

def clean_affordability():
    x = pd.read_excel(RAW / "affordability_ratios.xlsx", "5c", header=1)
    x.columns = [str(c).split(".")[0] for c in x.columns]
    x = x[x["Country/Region name"] == "London"]
    years = [c for c in x.columns if c.isdigit()]
    a = x.melt(id_vars=["Local authority code", "Local authority name"], value_vars=years, var_name="year", value_name="ratio")
    a.columns = ["area_code", "area", "year", "ratio"]
    reg = pd.read_excel(RAW / "affordability_ratios.xlsx", "1c", header=1)
    reg.columns = [str(c).split(".")[0] for c in reg.columns]
    reg = reg[reg.Name.isin(["London", "England"])].melt(id_vars=["Code", "Name"], value_vars=years, var_name="year", value_name="ratio")
    reg.columns = ["area_code", "area", "year", "ratio"]
    a = pd.concat([a, reg])
    a["year"] = a.year.astype(int); a["ratio"] = pd.to_numeric(a.ratio, errors="coerce")
    a.to_csv(PROC / "affordability.csv", index=False)
    return a

def clean_ppd(hpi):
    p = pd.concat(pd.read_csv(f) for f in sorted(RAW.glob("pp_london_*.csv")))
    n0 = len(p)
    # Standard market-value sales only (category A) of the four main property types
    p = p[(p.ppd_category == "A") & p.property_type.isin(TYPES) & p.postcode.notna()].copy()
    p["date"] = pd.to_datetime(p.date)
    p["property_type"] = p.property_type.map(TYPES)
    p["tenure"] = p.tenure.map({"F": "Freehold", "L": "Leasehold"})
    p["new_build"] = (p.new_build == "Y").astype(int)
    p["postcode_district"] = p.postcode.str.split(" ").str[0]
    names = hpi.drop_duplicates("area_code").set_index(hpi.drop_duplicates("area_code").area.str.upper()).area
    p["borough"] = p.district.map(names)
    p = p.dropna(subset=["tenure", "borough"])
    p = p[["transaction_id", "date", "price", "property_type", "tenure", "new_build", "borough", "postcode_district"]]
    p.sort_values("date").to_csv(INT / "ppd_london.csv", index=False)
    print(f"PPD: {n0:,} London rows -> {len(p):,} standard residential sales ({p.date.min():%d %b %Y} to {p.date.max():%d %b %Y})")
    return p

if __name__ == "__main__":
    h = clean_hpi(); a = clean_affordability(); p = clean_ppd(h)
    print(f"HPI: {h.area.nunique()} areas, {h.month.min():%b %Y} to {h.month.max():%b %Y}; affordability: {a.area.nunique()} areas, {a.year.min()}-{a.year.max()}")

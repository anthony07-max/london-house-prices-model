"""Download public housing data into data/raw/.

Sources (all Open Government Licence v3.0)
- HM Land Registry Price Paid Data, yearly files 2023 to 2026 (year to date). Only Greater London rows are kept.
  https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
- HM Land Registry UK House Price Index, full file (monthly, by local authority, from 1995 for London).
  https://www.gov.uk/government/collections/uk-house-price-index-reports
- ONS House price to workplace-based earnings ratio (affordability), local authority tables.
  https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/ratioofhousepricetoworkplacebasedearningslowerquartileandmedian
"""
import datetime as dt, pathlib, tempfile
import pandas as pd, requests

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio project; UK open data)"}
PPD = "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}.csv"
YEARS = [2023, 2024, 2025, 2026]
PPD_COLS = ["transaction_id", "price", "date", "postcode", "property_type", "new_build", "tenure", "paon", "saon",
            "street", "locality", "town", "district", "county", "ppd_category", "record_status"]
HPI = "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-{ym}.csv"
AFF = ("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/housing/datasets/"
       "ratioofhousepricetoworkplacebasedearningslowerquartileandmedian/current/aff1ratioofhousepricetoworkplacebasedearnings.xlsx")

def fetch(url, dest, stream=False):
    with requests.get(url, headers=HEADERS, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        out = RAW / f"pp_london_{year}.csv"
        print("Downloading", PPD.format(year=year))
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            fetch(PPD.format(year=year), tmp.name)
            # Keep only the columns we need; addresses (house number, street) are dropped here.
            keep = ["transaction_id", "price", "date", "postcode", "property_type", "new_build", "tenure",
                    "district", "county", "ppd_category", "record_status"]
            parts = [c[c.county == "GREATER LONDON"] for c in
                     pd.read_csv(tmp.name, header=None, names=PPD_COLS, usecols=keep, chunksize=250_000)]
            london = pd.concat(parts)
            london.to_csv(out, index=False)
            print(f"  kept {len(london):,} Greater London transactions")

    # Latest UK HPI full file: try this month, then step back until one exists.
    d = dt.date.today().replace(day=1)
    for _ in range(6):
        ym = f"{d.year}-{d.month:02d}"
        r = requests.head(HPI.format(ym=ym), headers=HEADERS, timeout=60)
        if r.status_code == 200:
            print("Downloading", HPI.format(ym=ym)); fetch(HPI.format(ym=ym), RAW / "uk_hpi_full.csv")
            break
        d = (d - dt.timedelta(days=1)).replace(day=1)
    else:
        raise RuntimeError("No UK HPI full file found for the last 6 months")

    print("Downloading", AFF); fetch(AFF, RAW / "affordability_ratios.xlsx")
    print("Done:", sorted(p.name for p in RAW.iterdir()))

if __name__ == "__main__":
    main()

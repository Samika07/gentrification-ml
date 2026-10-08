import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("CENSUS_API_KEY")

if not API_KEY:
    raise ValueError(
        "CENSUS_API_KEY not found in .env"
    )

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STATE = "06"  # California


# ============================================================
# VARIABLES WE WOULD LIKE TO USE
# ============================================================

REQUESTED_VARIABLES = {

    "B01003_001E": "population",

    "B01002_001E": "median_age",

    "B19013_001E": "median_household_income",

    "B17001_001E": "poverty_universe",

    "B17001_002E": "poverty_below",

    "B15003_001E": "education_total",

    "B15003_022E": "bachelors",

    "B15003_023E": "masters",

    "B15003_024E": "professional_degree",

    "B15003_025E": "doctorate",

    "B25001_001E": "housing_units",

    "B25064_001E": "median_gross_rent",

    "B25077_001E": "median_home_value",
}


# ============================================================
# CHECK WHETHER A VARIABLE EXISTS
# ============================================================

def get_available_variables(year):

    print(
        f"Checking available variables for {year}..."
    )

    url = (
        f"https://api.census.gov/data/"
        f"{year}/acs/acs5/variables.json"
    )

    response = requests.get(
        url,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Could not retrieve variable list for {year}"
        )

    return response.json()["variables"]


# ============================================================
# FILTER VARIABLES
# ============================================================

def get_valid_variables(year):

    available = get_available_variables(year)

    valid = {}
    missing = {}

    for census_name, clean_name in REQUESTED_VARIABLES.items():

        if census_name in available:

            valid[census_name] = clean_name

        else:

            missing[census_name] = clean_name

    print()
    print("Valid variables:")

    for variable in valid:
        print(
            f"  {variable} -> {valid[variable]}"
        )

    if missing:

        print()
        print("Variables unavailable for this year:")

        for variable in missing:
            print(
                f"  {variable} -> {missing[variable]}"
            )

    return valid


# ============================================================
# DOWNLOAD ONE YEAR
# ============================================================

def download_year(year):

    print()
    print("=" * 70)
    print(
        f"Downloading ACS 5-Year data for {year}"
    )
    print("=" * 70)

    variables = get_valid_variables(year)

    if not variables:

        raise RuntimeError(
            f"No valid variables found for {year}"
        )

    request_variables = [
        "NAME"
    ] + list(variables.keys())

    get_string = ",".join(
        request_variables
    )

    url = (
        f"https://api.census.gov/data/"
        f"{year}/acs/acs5"
        f"?get={get_string}"
        f"&for=tract:*"
        f"&in=state:{STATE}"
        f"&key={API_KEY}"
    )

    print()
    print("Requesting Census API...")

    response = requests.get(
        url,
        timeout=120
    )

    print(
        "HTTP status:",
        response.status_code
    )

    if response.status_code != 200:

        print(response.text)

        raise RuntimeError(
            f"Census request failed for {year}"
        )

    try:

        data = response.json()

    except ValueError:

        print(
            response.text[:2000]
        )

        raise RuntimeError(
            f"Census returned invalid JSON for {year}"
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        data[1:],
        columns=data[0]
    )

    # ========================================================
    # RENAME
    # ========================================================

    df = df.rename(
        columns=variables
    )

    # ========================================================
    # TRACT ID
    # ========================================================

    df["TRACT_ID"] = (
        df["state"].astype(str)
        + "_"
        + df["county"].astype(str)
        + "_"
        + df["tract"].astype(str)
    )

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    for column in variables.values():

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    if (
        "poverty_below" in df.columns
        and
        "poverty_universe" in df.columns
    ):

        df["poverty_pct"] = (
            df["poverty_below"]
            /
            df["poverty_universe"].replace(
                0,
                pd.NA
            )
        ) * 100

    else:

        df["poverty_pct"] = pd.NA


    if (
        "education_total" in df.columns
        and
        "bachelors" in df.columns
        and
        "masters" in df.columns
        and
        "professional_degree" in df.columns
        and
        "doctorate" in df.columns
    ):

        df["higher_education_pct"] = (
            (
                df["bachelors"]
                +
                df["masters"]
                +
                df["professional_degree"]
                +
                df["doctorate"]
            )
            /
            df["education_total"].replace(
                0,
                pd.NA
            )
        ) * 100

    else:

        df["higher_education_pct"] = pd.NA


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        RAW_DIR
        /
        f"ca_{year}.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    YEARS = [
        2010,
        2011,
        2012,
        2016
    ]

    for year in YEARS:

        download_year(year)

    print()
    print("=" * 70)
    print(
        "ALL DOWNLOADS COMPLETED"
    )
    print("=" * 70)
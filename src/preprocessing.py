import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_year(year):

    path = RAW_DIR / f"ca_{year}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    print(
        f"{year}: "
        f"{df.shape[0]} rows x "
        f"{df.shape[1]} columns"
    )

    return df


# ============================================================
# BASIC CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    # Remove duplicate Census tracts
    df = df.drop_duplicates(
        subset="TRACT_ID"
    )

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# CREATE 2010 -> 2011 FEATURES
# ============================================================

def create_features(df_2010, df_2011):

    # --------------------------------------------------------
    # Only use variables that actually exist in BOTH years.
    #
    # Education is deliberately excluded because the 2010
    # and 2011 files do not contain those variables.
    # --------------------------------------------------------

    features = [
        "population",
        "median_age",
        "median_household_income",
        "poverty_pct",
        "housing_units",
        "median_gross_rent",
        "median_home_value"
    ]

    # Keep only the required columns
    df10 = df_2010[
        ["TRACT_ID"] + features
    ].copy()

    df11 = df_2011[
        ["TRACT_ID"] + features
    ].copy()

    # Merge the same Census tracts
    merged = df10.merge(
        df11,
        on="TRACT_ID",
        how="inner",
        suffixes=(
            "_2010",
            "_2011"
        )
    )

    print(
        f"\nCommon 2010-2011 tracts: "
        f"{len(merged)}"
    )

    # --------------------------------------------------------
    # Calculate percentage / absolute changes
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["TRACT_ID"] = (
        merged["TRACT_ID"]
    )

    # Population percentage change
    result["population_change_pct"] = (
        (
            merged["population_2011"]
            -
            merged["population_2010"]
        )
        /
        merged["population_2010"].replace(
            0,
            np.nan
        )
    ) * 100

    # Median age change
    result["median_age_change"] = (
        merged["median_age_2011"]
        -
        merged["median_age_2010"]
    )

    # Income percentage change
    result["income_change_pct"] = (
        (
            merged["median_household_income_2011"]
            -
            merged["median_household_income_2010"]
        )
        /
        merged[
            "median_household_income_2010"
        ].replace(
            0,
            np.nan
        )
    ) * 100

    # Poverty percentage change
    result["poverty_change"] = (
        merged["poverty_pct_2011"]
        -
        merged["poverty_pct_2010"]
    )

    # Housing unit percentage change
    result["housing_units_change_pct"] = (
        (
            merged["housing_units_2011"]
            -
            merged["housing_units_2010"]
        )
        /
        merged["housing_units_2010"].replace(
            0,
            np.nan
        )
    ) * 100

    # Rent percentage change
    result["rent_change_pct"] = (
        (
            merged["median_gross_rent_2011"]
            -
            merged["median_gross_rent_2010"]
        )
        /
        merged[
            "median_gross_rent_2010"
        ].replace(
            0,
            np.nan
        )
    ) * 100

    # Home value percentage change
    result["home_value_change_pct"] = (
        (
            merged["median_home_value_2011"]
            -
            merged["median_home_value_2010"]
        )
        /
        merged[
            "median_home_value_2010"
        ].replace(
            0,
            np.nan
        )
    ) * 100

    return result


# ============================================================
# EXTRA FEATURE
# ============================================================

def create_extra_feature(df_2011):

    df = df_2011[
        [
            "TRACT_ID",
            "median_gross_rent",
            "median_household_income"
        ]
    ].copy()

    # Convert annual household income to monthly income
    monthly_income = (
        df["median_household_income"]
        /
        12
    )

    # --------------------------------------------------------
    # EXTRA FEATURE:
    #
    # Housing affordability pressure
    #
    # median monthly rent
    # -------------------
    # monthly median income
    # --------------------------------------------------------

    df["affordability_pressure"] = (
        df["median_gross_rent"]
        /
        monthly_income.replace(
            0,
            np.nan
        )
    )

    return df[
        [
            "TRACT_ID",
            "affordability_pressure"
        ]
    ]


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(df_2012, df_2016):

    df12 = df_2012[
        [
            "TRACT_ID",
            "median_gross_rent"
        ]
    ].copy()

    df16 = df_2016[
        [
            "TRACT_ID",
            "median_gross_rent"
        ]
    ].copy()

    # Merge common Census tracts
    merged = df12.merge(
        df16,
        on="TRACT_ID",
        how="inner",
        suffixes=(
            "_2012",
            "_2016"
        )
    )

    print(
        f"Common 2012-2016 tracts: "
        f"{len(merged)}"
    )

    # --------------------------------------------------------
    # Housing cost change
    # --------------------------------------------------------

    merged[
        "housing_cost_change_pct"
    ] = (
        (
            merged["median_gross_rent_2016"]
            -
            merged["median_gross_rent_2012"]
        )
        /
        merged[
            "median_gross_rent_2012"
        ].replace(
            0,
            np.nan
        )
    ) * 100

    # --------------------------------------------------------
    # Binary target
    #
    # 1 = >= 10% increase
    # 0 = < 10% increase
    #
    # IMPORTANT:
    # This is our project threshold, not claimed to be the
    # exact original Stanford labeling rule.
    # --------------------------------------------------------

    merged["gentrification"] = (
        merged[
            "housing_cost_change_pct"
        ]
        >= 10
    ).astype(int)

    return merged[
        [
            "TRACT_ID",
            "housing_cost_change_pct",
            "gentrification"
        ]
    ]


# ============================================================
# BUILD FINAL DATASET
# ============================================================

def build_dataset():

    print("\n" + "=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    df2010 = load_year(2010)
    df2011 = load_year(2011)
    df2012 = load_year(2012)
    df2016 = load_year(2016)

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    df2010 = clean_data(df2010)
    df2011 = clean_data(df2011)
    df2012 = clean_data(df2012)
    df2016 = clean_data(df2016)

    # --------------------------------------------------------
    # Create 2010 -> 2011 features
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING TEMPORAL FEATURES")
    print("=" * 70)

    features = create_features(
        df2010,
        df2011
    )

    # --------------------------------------------------------
    # Create extra feature
    # --------------------------------------------------------

    print("\nCreating extra feature...")

    extra = create_extra_feature(
        df2011
    )

    # --------------------------------------------------------
    # Create target
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING TARGET")
    print("=" * 70)

    target = create_target(
        df2012,
        df2016
    )

    # --------------------------------------------------------
    # Merge features + extra feature
    # --------------------------------------------------------

    final = features.merge(
        extra,
        on="TRACT_ID",
        how="inner"
    )

    # --------------------------------------------------------
    # Merge target
    # --------------------------------------------------------

    final = final.merge(
        target,
        on="TRACT_ID",
        how="inner"
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    final = final.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Remove rows where required values are missing
    # --------------------------------------------------------

    feature_columns = [
        "population_change_pct",
        "median_age_change",
        "income_change_pct",
        "poverty_change",
        "housing_units_change_pct",
        "rent_change_pct",
        "home_value_change_pct",
        "affordability_pressure"
    ]

    final = final.dropna(
        subset=feature_columns + [
            "gentrification"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR
        /
        "gentrification_dataset.csv"
    )

    final.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal dataset shape: "
        f"{final.shape}"
    )

    print(
        "\nFeatures:"
    )

    for column in feature_columns:
        print(
            f"  - {column}"
        )

    print(
        "\nTarget:"
    )

    print(
        "  - gentrification"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        final[
            "gentrification"
        ].value_counts()
    )

    print(
        "\nMissing values:"
    )

    print(
        final.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        f"\nSaved to:"
        f"\n{output_path}"
    )

    return final


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_dataset()
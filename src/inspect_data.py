import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("data/raw")

YEARS = [
    2010,
    2011,
    2012,
    2016
]


# ============================================================
# INSPECT ONE DATASET
# ============================================================

def inspect_year(year):

    path = RAW_DIR / f"ca_{year}.csv"

    print()
    print("=" * 70)
    print(f"DATASET: {year}")
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    print("\nShape:")
    print(
        f"Rows    : {df.shape[0]}"
    )
    print(
        f"Columns : {df.shape[1]}"
    )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print("\nColumns:")

    for i, column in enumerate(df.columns, start=1):
        print(
            f"{i:2}. {column}"
        )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\nData types:")

    print(
        df.dtypes.to_string()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:
        print("No missing values.")

    else:
        print(
            missing.to_string()
        )

    # --------------------------------------------------------
    # Duplicate tract IDs
    # --------------------------------------------------------

    if "TRACT_ID" in df.columns:

        duplicates = (
            df["TRACT_ID"]
            .duplicated()
            .sum()
        )

        print(
            f"\nDuplicate TRACT_IDs: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # First 5 rows
    # --------------------------------------------------------

    print("\nFirst 5 rows:")

    print(
        df.head().to_string()
    )

    # --------------------------------------------------------
    # Numerical summary
    # --------------------------------------------------------

    print("\nNumerical summary:")

    print(
        df.describe()
        .round(2)
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CENSUS DATA INSPECTION")
    print("=" * 70)

    for year in YEARS:
        inspect_year(year)

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
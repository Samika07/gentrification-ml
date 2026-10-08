# Gentrification Prediction using Machine Learning

## Project Overview

This project aims to predict gentrification at the U.S. Census tract level using demographic, economic, and housing-related data from the U.S. Census American Community Survey (ACS).

The current pipeline covers:

1. Census data collection
2. Data inspection
3. Data cleaning
4. Feature engineering
5. Creation of the gentrification target
6. Generation of an ML-ready dataset

The next stage is machine learning model training and evaluation.

---

## Dataset

The project uses California Census tract-level ACS 5-Year data for:

- 2010
- 2011
- 2012
- 2016

Raw datasets are stored in:

```text
data/raw/
├── ca_2010.csv
├── ca_2011.csv
├── ca_2012.csv
└── ca_2016.csv

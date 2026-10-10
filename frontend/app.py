"""Streamlit frontend for the Ames house-price prediction pipeline."""

from __future__ import annotations

import calendar
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Running ``streamlit run frontend/app.py`` makes ``frontend`` the script
# directory. Add the project root so the sibling backend package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.backend import (  # noqa: E402
    ENGINEERED_FEATURES,
    category_options,
    default_values,
    load_model,
    market_percentile,
    model_features,
    predict_price,
    reference_price_stats,
)


st.set_page_config(
    page_title="Haven | House price estimator",
    page_icon=":material/home_work:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Friendly names are displayed with the original Ames code in parentheses.
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "MSZoning": {
        "A": "Agriculture", "C (all)": "Commercial", "FV": "Floating village residential",
        "I": "Industrial", "RH": "Residential high density", "RL": "Residential low density",
        "RP": "Residential low-density park", "RM": "Residential medium density",
    },
    "Neighborhood": {
        "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
        "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
        "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
        "IDOTRR": "Iowa DOT and Rail Road", "MeadowV": "Meadow Village",
        "Mitchel": "Mitchell", "NAmes": "North Ames", "NoRidge": "Northridge",
        "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
        "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South/West ISU",
        "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset",
        "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
    },
    "HouseStyle": {
        "1Story": "One story", "1.5Fin": "One-and-a-half story, finished",
        "1.5Unf": "One-and-a-half story, unfinished", "2Story": "Two story",
        "2.5Fin": "Two-and-a-half story, finished", "2.5Unf": "Two-and-a-half story, unfinished",
        "SFoyer": "Split foyer", "SLvl": "Split level",
    },
    "BldgType": {
        "1Fam": "Single-family detached", "2fmCon": "Two-family conversion",
        "Duplex": "Duplex", "Twnhs": "Townhouse", "TwnhsE": "Townhouse end unit",
    },
    "GarageType": {
        "2Types": "More than one garage type", "Attchd": "Attached garage",
        "Basment": "Basement garage", "BuiltIn": "Built-in garage", "CarPort": "Carport",
        "Detchd": "Detached garage", "None": "No garage",
    },
    "GarageFinish": {
        "Fin": "Finished", "RFn": "Rough finished", "Unf": "Unfinished", "None": "No garage",
    },
    "SaleType": {
        "WD": "Normal warranty deed", "CWD": "Cash warranty deed", "VWD": "VA warranty deed",
        "New": "New home", "COD": "Court officer deed", "Con": "Contract",
        "ConLw": "Low down-payment contract", "ConLI": "Low-interest contract",
        "ConLD": "Low down-payment/interest contract", "Oth": "Other",
    },
    "SaleCondition": {
        "Normal": "Normal sale", "Abnorml": "Abnormal sale", "AdjLand": "Adjoining land purchase",
        "Alloca": "Allocation sale", "Family": "Sale between family members",
        "Partial": "Partial or unfinished home sale",
    },
    "CentralAir": {"Y": "Yes", "N": "No"},
    "PavedDrive": {"Y": "Paved", "P": "Partially paved", "N": "Unpaved"},
}

QUALITY_LABELS = {
    "Ex": "Excellent", "Gd": "Good", "TA": "Typical/average",
    "Fa": "Fair", "Po": "Poor", "None": "Not present",
}
for quality_column in ("ExterQual", "KitchenQual", "HeatingQC", "BsmtQual", "FireplaceQu"):
    CATEGORY_LABELS[quality_column] = QUALITY_LABELS


FIELD_GUIDE = [
    ("Neighborhood", "Neighborhood", "Physical neighborhood within Ames, Iowa", "category"),
    ("Zoning classification", "MSZoning", "General zoning classification of the sale", "category"),
    ("Building type", "BldgType", "Detached home, duplex, townhouse, and similar types", "category"),
    ("House style", "HouseStyle", "Number of stories and dwelling layout", "category"),
    ("Year built", "YearBuilt", "Original construction year", "year"),
    ("Year remodeled", "YearRemodAdd", "Most recent remodel year; use build year if never remodeled", "year"),
    ("Living area", "GrLivArea", "Finished living area above ground", "square feet"),
    ("Street frontage", "LotFrontage", "Street length directly connected to the property", "linear feet"),
    ("Lot area", "LotArea", "Total land area", "square feet"),
    ("Basement area", "TotalBsmtSF", "Total basement floor area", "square feet"),
    ("First-floor area", "1stFlrSF", "Area of the first floor", "square feet"),
    ("Second-floor area", "2ndFlrSF", "Area of the second floor", "square feet"),
    ("Material and finish quality", "OverallQual", "Overall quality: 1 very poor to 10 excellent", "1–10 rating"),
    ("Overall condition", "OverallCond", "Present condition: 1 very poor to 10 excellent", "1–10 rating"),
    ("Exterior quality", "ExterQual", "Quality of exterior materials", "quality grade"),
    ("Basement quality", "BsmtQual", "Basement height/quality grade", "quality grade"),
    ("Heating quality", "HeatingQC", "Heating-system quality and condition", "quality grade"),
    ("Kitchen quality", "KitchenQual", "Kitchen quality rating", "quality grade"),
    ("Bedrooms", "BedroomAbvGr", "Bedrooms above basement level", "count"),
    ("Kitchens", "KitchenAbvGr", "Kitchens above basement level", "count"),
    ("Rooms above ground", "TotRmsAbvGrd", "Rooms above ground, excluding bathrooms", "count"),
    ("Full bathrooms", "FullBath", "Full bathrooms above ground", "count"),
    ("Half bathrooms", "HalfBath", "Half bathrooms above ground", "count"),
    ("Basement full bathrooms", "BsmtFullBath", "Full bathrooms in the basement", "count"),
    ("Basement half bathrooms", "BsmtHalfBath", "Half bathrooms in the basement", "count"),
    ("Fireplaces", "Fireplaces", "Number of fireplaces", "count"),
    ("Fireplace quality", "FireplaceQu", "Fireplace quality grade", "quality grade"),
    ("Garage type", "GarageType", "Garage location/design", "category"),
    ("Garage finish", "GarageFinish", "Interior finish level of the garage", "category"),
    ("Garage capacity", "GarageCars", "Number of cars the garage holds", "cars"),
    ("Garage area", "GarageArea", "Interior garage area", "square feet"),
    ("Driveway surface", "PavedDrive", "Whether the driveway is paved", "category"),
    ("Wood deck", "WoodDeckSF", "Wood deck area", "square feet"),
    ("Open porch", "OpenPorchSF", "Open porch area", "square feet"),
    ("Enclosed porch", "EnclosedPorch", "Enclosed porch area", "square feet"),
    ("Screen porch", "ScreenPorch", "Screened porch area", "square feet"),
    ("Pool area", "PoolArea", "Pool area; zero means no pool", "square feet"),
    ("Central air", "CentralAir", "Whether central air conditioning is installed", "yes/no"),
    ("Sale type", "SaleType", "Type of deed or sale transaction", "category"),
    ("Sale condition", "SaleCondition", "Circumstances of the sale", "category"),
    ("Sale year", "YrSold", "Calendar year when the property was sold", "year"),
    ("Sale month", "MoSold", "Calendar month when the property was sold", "month"),
]


def friendly_option(column: str, value: str) -> str:
    code = str(value)
    friendly = CATEGORY_LABELS.get(column, {}).get(code)
    return f"{friendly} ({code})" if friendly else code


def option_index(options: list[str], value: str) -> int:
    try:
        return options.index(str(value))
    except ValueError:
        return 0


def select_category(label: str, column: str, help_text: str):
    options = category_options(column)
    return st.selectbox(
        label,
        options,
        index=option_index(options, defaults[column]),
        format_func=lambda value: friendly_option(column, value),
        help=f"{help_text} Dataset column: {column}.",
        key=f"input_{column}",
    )


@st.cache_resource
def ensure_model_is_ready():
    return load_model()


try:
    ensure_model_is_ready()
    defaults = default_values()
except Exception as exc:
    st.error(f"Could not initialize the prediction model: {exc}", icon=":material/error:")
    st.stop()


with st.sidebar:
    st.title("Haven", anchor=False)
    st.caption("Ames, Iowa house-price model")
    st.info(
        "Hover over the **?** beside any field to see its definition and original dataset column name.",
        icon=":material/help:",
    )
    stats = reference_price_stats()
    st.metric("Training-set median price", f"${stats['median']:,.0f}")
    st.metric("Model input columns", len(model_features()))
    st.caption(
        "The form exposes understandable property details. Other technical fields use "
        "the median or most-common value from the cleaned training data."
    )

st.title("House price estimator", icon=":material/home_work:")
st.markdown(
    "Enter the property's details below. Every label uses plain English, and the help icon "
    "shows the exact Ames Housing dataset column used by the model."
)

estimator_tab, guide_tab, process_tab = st.tabs(
    [":material/calculate: Estimate", ":material/menu_book: Field guide", ":material/account_tree: How it works"]
)

with estimator_tab:
    st.caption("Start with the main facts. Open the optional sections for a more property-specific estimate.")
    with st.form("prediction_form"):
        st.subheader("Main property details", icon=":material/home:")
        location_col, size_col, quality_col = st.columns(3)

        with location_col:
            neighborhood = select_category("Neighborhood", "Neighborhood", "Physical neighborhood in Ames, Iowa.")
            zoning = select_category("Zoning classification", "MSZoning", "General zoning classification.")
            building_type = select_category("Building type", "BldgType", "Detached home, duplex, or townhouse.")
            house_style = select_category("House style", "HouseStyle", "Number of stories and dwelling layout.")
            year_built = st.number_input(
                "Year built", 1870, 2010, int(defaults["YearBuilt"]), step=1,
                help="Original construction year. Dataset column: YearBuilt.",
            )
            year_remodeled = st.number_input(
                "Year remodeled", 1870, 2010, int(defaults["YearRemodAdd"]), step=1,
                help="Use the build year if never remodeled. Dataset column: YearRemodAdd.",
            )

        with size_col:
            living_area = st.number_input(
                "Living area above ground (sq ft)", 300, 6000, int(defaults["GrLivArea"]), step=50,
                help="Finished living area above basement level. Dataset column: GrLivArea.",
            )
            lot_frontage = st.number_input(
                "Street frontage (linear ft)", 20, 350, int(defaults["LotFrontage"]), step=5,
                help="Street length connected to the property. Dataset column: LotFrontage.",
            )
            lot_area = st.number_input(
                "Lot area (sq ft)", 1000, 220000, int(defaults["LotArea"]), step=100,
                help="Total land area. Dataset column: LotArea.",
            )
            basement_area = st.number_input(
                "Total basement area (sq ft)", 0, 6500, int(defaults["TotalBsmtSF"]), step=50,
                help="Enter 0 for no basement. Dataset column: TotalBsmtSF.",
            )
            first_floor = st.number_input(
                "First-floor area (sq ft)", 300, 5000, int(defaults["1stFlrSF"]), step=50,
                help="Dataset column: 1stFlrSF.",
            )
            second_floor = st.number_input(
                "Second-floor area (sq ft)", 0, 2500, int(defaults["2ndFlrSF"]), step=50,
                help="Enter 0 for a one-story home. Dataset column: 2ndFlrSF.",
            )

        with quality_col:
            overall_quality = st.slider(
                "Material and finish quality", 1, 10, int(defaults["OverallQual"]),
                help="1 = very poor; 10 = excellent. Dataset column: OverallQual.",
            )
            overall_condition = st.slider(
                "Overall condition", 1, 10, int(defaults["OverallCond"]),
                help="1 = very poor; 10 = excellent. Dataset column: OverallCond.",
            )
            exterior_quality = select_category("Exterior material quality", "ExterQual", "Exterior material quality.")
            basement_quality = select_category("Basement quality", "BsmtQual", "Basement height/quality grade.")
            heating_quality = select_category("Heating system quality", "HeatingQC", "Heating quality and condition.")
            kitchen_quality = select_category("Kitchen quality", "KitchenQual", "Overall kitchen quality.")

        with st.expander("Rooms and bathrooms", icon=":material/bed:"):
            room_a, room_b, room_c = st.columns(3)
            with room_a:
                bedrooms = st.number_input(
                    "Bedrooms above basement", 0, 8, int(defaults["BedroomAbvGr"]), step=1,
                    help="Dataset column: BedroomAbvGr.",
                )
                kitchens = st.number_input(
                    "Kitchens above basement", 0, 3, int(defaults["KitchenAbvGr"]), step=1,
                    help="Dataset column: KitchenAbvGr.",
                )
                total_rooms = st.number_input(
                    "Total rooms above ground", 2, 15, int(defaults["TotRmsAbvGrd"]), step=1,
                    help="Excludes bathrooms. Dataset column: TotRmsAbvGrd.",
                )
            with room_b:
                full_bath = st.number_input("Full bathrooms above ground", 0, 4, int(defaults["FullBath"]), step=1, help="Dataset column: FullBath.")
                half_bath = st.number_input("Half bathrooms above ground", 0, 3, int(defaults["HalfBath"]), step=1, help="Counts as 0.5. Dataset column: HalfBath.")
            with room_c:
                basement_full_bath = st.number_input("Basement full bathrooms", 0, 3, int(defaults["BsmtFullBath"]), step=1, help="Dataset column: BsmtFullBath.")
                basement_half_bath = st.number_input("Basement half bathrooms", 0, 2, int(defaults["BsmtHalfBath"]), step=1, help="Counts as 0.5. Dataset column: BsmtHalfBath.")

        with st.expander("Garage, fireplace and outdoor areas", icon=":material/garage_home:"):
            st.caption(
                "If an amenity is absent, set its count and area to 0. During training, missing "
                "quality/type labels were filled with the most-common category by the preprocessor."
            )
            amenity_a, amenity_b, amenity_c = st.columns(3)
            with amenity_a:
                garage_type = select_category("Garage type", "GarageType", "Garage location/design.")
                garage_finish = select_category("Garage finish", "GarageFinish", "Interior finish level.")
                garage_cars = st.number_input("Garage capacity (cars)", 0, 5, int(defaults["GarageCars"]), step=1, help="Dataset column: GarageCars.")
                garage_area = st.number_input("Garage area (sq ft)", 0, 1500, int(defaults["GarageArea"]), step=25, help="Dataset column: GarageArea.")
                paved_drive = select_category("Driveway surface", "PavedDrive", "Whether the driveway is paved.")
            with amenity_b:
                fireplaces = st.number_input("Number of fireplaces", 0, 4, int(defaults["Fireplaces"]), step=1, help="Dataset column: Fireplaces.")
                fireplace_quality = select_category("Fireplace quality", "FireplaceQu", "Fireplace quality grade.")
                central_air = select_category("Central air conditioning", "CentralAir", "Whether central air is installed.")
                wood_deck = st.number_input("Wood deck area (sq ft)", 0, 900, int(defaults["WoodDeckSF"]), step=10, help="Dataset column: WoodDeckSF.")
            with amenity_c:
                open_porch = st.number_input("Open porch area (sq ft)", 0, 750, int(defaults["OpenPorchSF"]), step=10, help="Dataset column: OpenPorchSF.")
                enclosed_porch = st.number_input("Enclosed porch area (sq ft)", 0, 600, int(defaults["EnclosedPorch"]), step=10, help="Dataset column: EnclosedPorch.")
                screen_porch = st.number_input("Screen porch area (sq ft)", 0, 500, int(defaults["ScreenPorch"]), step=10, help="Dataset column: ScreenPorch.")
                pool_area = st.number_input("Pool area (sq ft)", 0, 800, int(defaults["PoolArea"]), step=10, help="Enter 0 for no pool. Dataset column: PoolArea.")

        with st.expander("Sale information", icon=":material/contract:"):
            sale_a, sale_b = st.columns(2)
            with sale_a:
                sale_type = select_category("Sale type", "SaleType", "Type of deed or sale transaction.")
                sale_condition = select_category("Sale condition", "SaleCondition", "Circumstances of the sale.")
            with sale_b:
                year_sold = st.number_input(
                    "Sale year", 2006, 2010, int(defaults["YrSold"]), step=1,
                    help="Training sales cover 2006–2010. Dataset column: YrSold.",
                )
                month_sold = st.selectbox(
                    "Sale month", list(range(1, 13)), index=int(defaults["MoSold"]) - 1,
                    format_func=lambda month: calendar.month_name[month],
                    help="Dataset column: MoSold.",
                )

        submitted = st.form_submit_button(
            "Estimate property value", icon=":material/calculate:", type="primary", width="stretch"
        )

    if submitted:
        payload = {
            "Neighborhood": neighborhood, "MSZoning": zoning, "BldgType": building_type,
            "HouseStyle": house_style, "YearBuilt": year_built, "YearRemodAdd": year_remodeled,
            "GrLivArea": living_area, "LotFrontage": lot_frontage, "LotArea": lot_area,
            "TotalBsmtSF": basement_area, "1stFlrSF": first_floor, "2ndFlrSF": second_floor,
            "OverallQual": overall_quality, "OverallCond": overall_condition,
            "ExterQual": exterior_quality, "BsmtQual": basement_quality,
            "HeatingQC": heating_quality, "KitchenQual": kitchen_quality,
            "BedroomAbvGr": bedrooms, "KitchenAbvGr": kitchens, "TotRmsAbvGrd": total_rooms,
            "FullBath": full_bath, "HalfBath": half_bath, "BsmtFullBath": basement_full_bath,
            "BsmtHalfBath": basement_half_bath, "GarageType": garage_type,
            "GarageFinish": garage_finish, "GarageCars": garage_cars, "GarageArea": garage_area,
            "PavedDrive": paved_drive, "Fireplaces": fireplaces, "FireplaceQu": fireplace_quality,
            "CentralAir": central_air, "WoodDeckSF": wood_deck, "OpenPorchSF": open_porch,
            "EnclosedPorch": enclosed_porch, "ScreenPorch": screen_porch, "PoolArea": pool_area,
            "SaleType": sale_type, "SaleCondition": sale_condition,
            "YrSold": year_sold, "MoSold": month_sold,
        }
        try:
            price = predict_price(payload)
            percentile = market_percentile(price)
            with st.container(border=True):
                st.subheader("Estimated sale price", icon=":material/paid:")
                result_a, result_b = st.columns(2)
                result_a.metric("Model estimate", f"${price:,.0f}")
                result_b.metric("Training-data percentile", f"{percentile:.0f}th")
                st.caption(
                    f"Higher than approximately {percentile:.0f}% of training sale prices. "
                    "This is a model estimate, not a professional appraisal."
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}", icon=":material/error:")

with guide_tab:
    st.header("Frontend field guide", icon=":material/menu_book:")
    st.markdown(
        "**Frontend label** is what you see in the app. **Dataset column** is the original "
        "column in `cleaned_data.csv` that is sent to the trained pipeline."
    )
    guide_df = pd.DataFrame(FIELD_GUIDE, columns=["Frontend label", "Dataset column", "Meaning", "Value/unit"])
    st.dataframe(
        guide_df, hide_index=True, width="stretch",
        alt="Every editable frontend field and its Ames dataset column",
    )

    with st.expander("Category-code legend", icon=":material/translate:"):
        st.markdown(
            "Dropdowns show a readable name and the original code. For example, "
            "**Residential low density (RL)** sends `RL` to `MSZoning`. Quality grades are "
            "**Ex = excellent, Gd = good, TA = typical/average, Fa = fair, Po = poor**."
        )

    with st.expander("All 84 model input columns", icon=":material/table_view:"):
        visible_columns = {row[1] for row in FIELD_GUIDE}
        all_fields = []
        for column in model_features():
            if column in ENGINEERED_FEATURES:
                source = "Calculated by backend"
            elif column in visible_columns:
                source = "Editable in frontend"
            else:
                source = "Automatic training-data default"
            all_fields.append({"Dataset column": column, "How supplied": source})
        st.dataframe(
            pd.DataFrame(all_fields), hide_index=True, width="stretch",
            alt="All model input columns and how each value is supplied",
        )
        st.caption(
            "Automatic numbers use the training-data median. Automatic categories use the "
            "most-common training-data value."
        )

with process_tab:
    st.header("How the prediction is made", icon=":material/account_tree:")
    st.markdown(
        """
        1. **You enter property details.** The form collects areas, quality, rooms, amenities,
           location, and sale information.
        2. **The backend completes the row.** `backend/backend.py` fills fields not shown in the form with
           the median for numbers and the most-common training value for categories.
        3. **Five features are calculated.** The same engineered features from the training notebook
           are recreated for this property.
        4. **The saved pipeline preprocesses the row.** It fills missing values, scales numbers,
           and one-hot encodes categories.
        5. **Gradient Boosting predicts the price.** The fitted model in `best_model.pkl` returns
           an estimated sale price in US dollars.
        """
    )
    st.subheader("Calculated feature formulas")
    st.code(
        """TotalBathrooms = FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
HouseAge = YrSold - YearBuilt
RemodelAge = YrSold - YearRemodAdd""",
        language=None,
    )
    st.warning(
        "The model was trained on Ames, Iowa sales from 2006–2010. It does not include today's "
        "local market changes, inspection findings, or every unique property feature.",
        icon=":material/warning:",
    )


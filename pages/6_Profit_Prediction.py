import streamlit as st
import pandas as pd
#import plotly.express as px
from fredapi import Fred
import pickle

st.set_page_config(page_title="Profit Prediction", layout="wide")

st.markdown("<h1 style='text-align: center;'>Profit Predictions</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 24;">

    **Profit Predictions**

    Choose a Product Group on the left to Begin.

    </div>
    """,
    unsafe_allow_html=True
)

# Get Latest Fed Data
FRED_API_KEY = st.secrets["FED_API"]

fred = Fred(api_key=FRED_API_KEY)

cpi_auc = fred.get_series('CPIAUCSL')
cpi_fe = fred.get_series('CPILFESL')
ppi = fred.get_series('WPSFD4111')
di = fred.get_series('DTWEXBGS')

cpi_auc_last_item = cpi_auc.iloc[-1]
cpi_fe_last_item = cpi_fe.iloc[-1]
ppi_last_item = ppi.iloc[-1]
di_last_item = di.iloc[-1]

@st.cache_data()
def load_dataset():
    df = pd.read_csv('./pages/clean_retail_data.csv')
    return df


df = load_dataset()

st.sidebar.header("Select Filters")

subgroup_options = sorted(df['subgroup'].unique())
selected_subgroup = st.sidebar.selectbox(
    label="Step 1: Choose a Group",
    options=subgroup_options
)

subgroup_df = df[df['subgroup'] == selected_subgroup].copy()

sku_options = sorted(subgroup_df['sku'].unique())

if not sku_options:
    st.warning(f"No SKUs found for Subgroup '{selected_subgroup}'.")
    st.sidebar.warning("No SKUs available for this Subgroup.")
    st.stop()
else:
    selected_sku = st.sidebar.selectbox(
        label="Step 2: Choose a SKU",
        options=sku_options
    )

price = st.sidebar.text_input(
    label="Step 3: Enter Price")

units_ordered = st.sidebar.text_input(
    label="Step 4: Enter Units"
)

ad_spend = st.sidebar.text_input(
    label="Step 5: Enter Advertising Spend"
)

managed_stock_level = st.sidebar.text_input(
    label="Step 6: Enter Managed Stock Level"
)

profit_margin = st.sidebar.text_input(
    label="Step 7: Enter Profit Margin"
)

avg_pricing = st.sidebar.text_input(
    label="Step 8: Enter Average Pricing"
)

avg_comp_pricing = st.sidebar.text_input(
    label="Step 9: Enter Average Competitor Pricing"
)

data_to_predict = [
    price,
    units_ordered,
    ad_spend,
    managed_stock_level,
    di_last_item,
    ppi_last_item,
    profit_margin,
    avg_pricing,
    avg_comp_pricing
]

filename = "final_model.pkl"
model = None

with open(f'./pages/{filename}', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict([data_to_predict])

st.text(f"Predicted Profit: ${prediction[0]:.2f}")
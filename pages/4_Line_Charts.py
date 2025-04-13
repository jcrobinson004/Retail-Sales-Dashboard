import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Retail Sales Line Charts", layout="wide")

st.title("Line Charts")

st.markdown(
    """
    <div style="font-size: 24;">

    **Welcome to the Retail Sales Line Charts page.**

    Choose a Product Group, SKU and Metric on the left to Visualize.

    </div>
    """,
    unsafe_allow_html=True
)


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

final_df = subgroup_df[subgroup_df['sku'] == selected_sku]

final_df = final_df.sort_values(by='salesdate')

selected_chart = st.sidebar.selectbox(
    label="Step 3: Choose the Metric you'd like to plot",
    options=["Sales", "Profit", "Units Ordered"]
)

st.subheader(f"{selected_chart} Performance for SKU: {selected_sku} (Group: {selected_subgroup})")

if not final_df.empty:
    # Define columns to plot
    x_col = 'salesdate'
    if selected_chart == "Sales":
        y_col = 'sales'
        chart_title = f"Daily {y_col.capitalize()} for {selected_sku}"
        y_axis_title = "Daily Sales Amount ($)"
    elif selected_chart == "Profit":
        y_col = 'profit'
        chart_title = f"Daily {y_col.capitalize()} for {selected_sku}"
        y_axis_title = "Profit ($)"
    else:
        y_col = 'unitsordered'
        chart_title = f"Daily Units Ordered for {selected_sku}"
        y_axis_title = "Units Ordered"

    fig = px.line(
        data_frame=final_df,
        x=x_col,
        y=y_col,
        markers=True,
        title=chart_title,
        labels={y_col: y_axis_title, x_col: "Date"}
    )

    fig.update_layout(yaxis_title=y_axis_title)
    fig.update_traces(line=dict(width=2.5))

    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox(f"Show data table for {selected_sku}"):
        st.dataframe(final_df)

else:
    st.warning(f"No data found for Subgroup '{selected_subgroup}' and SKU '{selected_sku}'.")

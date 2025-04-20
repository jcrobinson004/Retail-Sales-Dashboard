import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Retail Sales Profits", layout="wide")

st.markdown("<h1 style='text-align: center;'>Bar Charts and Box Plots</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size: 24;">

    **Welcome to the Retail Sales Analysis page.**

    Choose a Product Group on the left to Visualize.

    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_data()
def load_dataset():
    df = pd.read_csv('./pages/clean_retail_data.csv')
    return df


df = load_dataset()

groups = list(df.subgroup.unique())
groups.sort()

group = st.sidebar.selectbox(
    'Choose a Product Group:',
    groups
)

if group != "All":
    group_df = df[df.subgroup == group]
else:
    group_df = df

# 1. Extract the number part using regex
#    - \d+ matches one or more digits
#    - expand=False returns a Series
group_df['sort_key'] = group_df['sku'].str.extract(r'(\d+)', expand=False)

# 2. Convert the extracted part to numeric (important!)
#    - errors='coerce' turns non-numeric results (like from 'sku') into NaN
group_df['sort_key'] = pd.to_numeric(group_df['sort_key'], errors='coerce')

# 3. Sort the DataFrame based on this numeric key
#    - na_position='last' puts items that couldn't be parsed at the end (or 'first')
group_df = group_df.sort_values(by='sort_key', na_position='last', ascending=False)

aggregation = group_df.groupby('sku')['profit'].agg(['mean', 'median']).reset_index()
melted_df = aggregation.melt(id_vars='sku', var_name='Statistic', value_name='Value')
sorted_groups_list = aggregation.sort_values(by='mean', ascending=False)['sku'].tolist()

if group == 'File Folders':
    chart_height = 1500
else:
    chart_height = 600

if not group_df.empty:
    boxplots = px.box(
        group_df,
        x="profit",
        y="sku",
        title=f"Box Plot of SKU vs. Profit for {group}",
        height=chart_height
    ).update_layout(
        title={
            'text': f"Box Plot of SKU vs. Profit for {group}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Profit ($) Per Day',
        yaxis_title='Individual SKU',
        font=dict(
            size=15,
        )
    )

    bar_chart = px.bar(
        melted_df,
        x="Value",
        y="sku",
        height=chart_height,
        color="Statistic",
        barmode="group",
        category_orders={
            "sku": sorted_groups_list
        }
    ).update_layout(
        title={
            'text': f'Mean and Median Profit Statistics by SKU for {group}',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Profit ($) Per Day',
        yaxis_title='Individual SKU',
        font=dict(
            size=15,
        )
    )

    # Group the data by genre and count the number of movies in each genre
    group_counts = df['subgroup'].value_counts().reset_index()
    group_counts.columns = ['Group', 'Count']

    treemap_viz = px.treemap(
        group_counts,
        path=['Group'],
        values='Count',
        # title='Movie Distribution by Genre',
        # color_continuous_scale=px.colors.sequential.RdBu,
        custom_data=["Group"]
    ).update_traces(
        marker=dict(
            cornerradius=5,
            colorscale='RdBu'
        ),
        textinfo="label+percent entry + value"
    ).update_layout(
        title={
            'text': 'Product Distribution by Group',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    st.plotly_chart(treemap_viz, use_container_width=False)
    st.plotly_chart(boxplots, use_container_width=True)
    st.plotly_chart(bar_chart, use_container_width=True)

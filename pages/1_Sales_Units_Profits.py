import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Retail Sales Profits", layout="wide")

st.title("Profits over Time")
st.markdown(
    """
    <div style="font-size: 24px;">

    **Welcome to the Retail Sales Profit page.**

    Choose a Product Group and Time Interval to Visualize Sales, Units, and Profits.

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
groups.insert(0, "All")

group = st.sidebar.selectbox(
    'Choose a Product Group:',
    groups,

)

time_select_list = [
    'Day',
    'Month',
    'Season'
]

time_selection = st.sidebar.selectbox(
    "Choose a Time Interval:",
    time_select_list
)

if time_selection == 'Day':
    column_to_select = "day_of_week"
elif time_selection == 'Month':
    column_to_select = "month"
else:
    column_to_select = "season"

if group != "All":
    group_df = df[df.subgroup == group]
else:
    group_df = df

if time_selection == 'Day':
    day_order = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

    agg_df = group_df.groupby('day_of_week', as_index=False)['profit'].mean()
    agg_df['day_of_week'] = pd.Categorical(agg_df['day_of_week'],
                                           categories=[m for m in day_order if m in agg_df['day_of_week'].unique()],
                                           ordered=True)
    agg_df = agg_df.sort_values('day_of_week')

    agg_df_units = group_df.groupby('day_of_week', as_index=False)['unitsordered'].mean()
    agg_df_units['day_of_week'] = pd.Categorical(agg_df_units['day_of_week'],
                                                 categories=[m for m in day_order if
                                                             m in agg_df_units['day_of_week'].unique()],
                                                 ordered=True)
    agg_df_units = agg_df_units.sort_values('day_of_week')

    agg_df_sales = group_df.groupby('day_of_week', as_index=False)['sales'].mean()
    agg_df_sales['day_of_week'] = pd.Categorical(agg_df_sales['day_of_week'],
                                                 categories=[m for m in day_order if
                                                             m in agg_df_sales['day_of_week'].unique()],
                                                 ordered=True)
    agg_df_sales = agg_df_sales.sort_values('day_of_week')
elif time_selection == 'Month':
    month_order = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December'
    ]
    agg_df = group_df.groupby('month', as_index=False)['profit'].mean()
    agg_df['profit_label'] = agg_df['profit'].apply(lambda x: f'${x:.2f}')
    agg_df['month'] = pd.Categorical(agg_df['month'],
                                     categories=[m for m in month_order if m in agg_df['month'].unique()],
                                     ordered=True)
    agg_df = agg_df.sort_values('month')

    agg_df_units = group_df.groupby('month', as_index=False)['unitsordered'].mean()
    agg_df_units['month'] = pd.Categorical(agg_df_units['month'],
                                                 categories=[m for m in month_order if
                                                             m in agg_df_units['month'].unique()],
                                                 ordered=True)
    agg_df_units = agg_df_units.sort_values('month')

    agg_df_sales = group_df.groupby('month', as_index=False)['sales'].mean()
    agg_df_sales['month'] = pd.Categorical(agg_df_sales['month'],
                                                 categories=[m for m in month_order if
                                                             m in agg_df_sales['month'].unique()],
                                                 ordered=True)
    agg_df_sales = agg_df_sales.sort_values('month')

else:
    season_order = [
        'Winter',
        'Spring',
        'Summer',
        'Fall'
    ]
    agg_df = group_df.groupby('season', as_index=False)['profit'].mean()
    agg_df['profit_label'] = agg_df['profit'].apply(lambda x: f'${x:.2f}')

    # Create a categorical type only with months present in the data to avoid errors
    agg_df['season'] = pd.Categorical(agg_df['season'],
                                      categories=[m for m in season_order if m in agg_df['season'].unique()],
                                      ordered=True)
    agg_df = agg_df.sort_values('season')  # Sort based on the defined order

    agg_df_units = group_df.groupby('season', as_index=False)['unitsordered'].mean()
    agg_df_units['month'] = pd.Categorical(agg_df_units['season'],
                                                 categories=[m for m in season_order if
                                                             m in agg_df_units['season'].unique()],
                                                 ordered=True)
    agg_df_units = agg_df_units.sort_values('season')

    agg_df_sales = group_df.groupby('season', as_index=False)['sales'].mean()
    agg_df_sales['season'] = pd.Categorical(agg_df_sales['season'],
                                                 categories=[m for m in season_order if
                                                             m in agg_df_sales['season'].unique()],
                                                 ordered=True)
    agg_df_sales = agg_df_sales.sort_values('season')

if not group_df.empty:
    line_chart_profit = px.line(
        data_frame=agg_df,
        x=column_to_select,
        y='profit',
        markers=True,
        # text='profit_label'
    ).update_layout(
        title={
            'text': f"Average Profit by {time_selection} for {group}",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f'{time_selection}',
        yaxis_title='Average Profit ($)',
        font=dict(
            size=15,
        )
    )

    line_chart_sales = px.line(
        data_frame=agg_df_sales,
        x=column_to_select,
        y='sales',
        markers=True,
        # text='profit_label'
    ).update_layout(
        title={
            'text': f"Average Sales by {time_selection} for {group}",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f'{time_selection}',
        yaxis_title='Average Sales ($)',
        font=dict(
            size=15,
        )
    )

    line_chart_units = px.line(
        data_frame=agg_df_sales,
        x=column_to_select,
        y='sales',
        markers=True,
        # text='profit_label'
    ).update_layout(
        title={
            'text': f"Average Units Ordered by {time_selection} for {group}",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f'{time_selection}',
        yaxis_title='Units Ordered',
        font=dict(
            size=15,
        )
    )

    st.plotly_chart(line_chart_sales, use_container_width=True)
    st.plotly_chart(line_chart_units, use_container_width=True)
    st.plotly_chart(line_chart_profit, use_container_width=True)


import streamlit as st
import pandas as pd

# Data Dictionary for Retail Sales Dashboard
# Set the title of the page
st.title("Data Dictionary")

# Add a description for the page
st.write("""
This page provides a detailed data dictionary for the Retail Sales Dashboard. 
Below, you will find descriptions of each variable used in the dataset.
""")
data_dictionary = {
    "sku": "This is the product ID, a unique identifier for each product.",
    "salesdate": "This represents the date on which a particular product was sold.",
    "price": "This is the price at which the product was sold on a given day.",
    "unitsordered": "This variable shows the number of units of a product ordered on a particular day.",
    "sales": "This represents the total revenue generated from the sale of a product on a given day (it is calculated as the product's price times the number of units ordered).",
    "cogs": "This stands for 'Cost of Goods Sold', which is the direct cost incurred by the company to produce or purchase the product.",
    "fba": "This is the eCommerce fee associated with selling the product. It includes the costs of storage, packing, and shipping handled by Amazon.",
    "reffee": "This is the eCommerce platform fee associated with selling the product (15% of sales).",
    "adspend": "This represents the advertisement cost associated with the product.",
    "profit": "This is the profit obtained from selling the product, calculated as sales minus the sum of cogs, fba, reffee, and adspend (profit = sales - cogs - fba - reffee - adspend).",
    "comp_x_price": "This represents the price of a similar product sold by a competitor. Up to 5 competitors' price data are available for each product (67 items have 0 competitors, 65 items have 1 competitor, 56 items have 2 competitors, 28 items have 3 competitors, 9 items have 4 competitors, 2 items have 5 competitors).",
    "comp_data_min_price": "This is the minimum price among all competitors for a similar product.",
    "comp_data_max_price": "This is the maximum price among all competitors for a similar product.",
    "managed_fba_stock_level": "This represents the available quantity of the product in stock.",
    "min_price": "This is the minimum allowable selling price for the product.",
    "max_price": "This is the maximum allowable selling price for the product."
}
# Convert the data dictionary to a pandas DataFrame
df_data_dictionary = pd.DataFrame(list(data_dictionary.items()), columns=["Variable", "Description"])

# Display the DataFrame in the Streamlit app
st.dataframe(df_data_dictionary)

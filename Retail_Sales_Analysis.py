import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Retail Sales Analysis",
    page_icon="🏠",
    layout="wide")

st.markdown("<h1 style='text-align: center;'>ADTA 5410 - Group 1 Weekly Challenge 4 - Retail Sales Analysis</h1>", unsafe_allow_html=True)

# Center the subheader using Markdown and HTML
st.markdown("<h3 style='text-align: center;'>Jordan Robinson, Kelly Welander, Ed Salinas</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Welcome to the Retail Analysis Page for Office Supplies</h4>", unsafe_allow_html=True)

try:
    office_supplies_one = Image.open('./images/office-supplies-1.png')
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(office_supplies_one, caption='Courtesy: Platinum Copier Solutions')


except FileNotFoundError:
    st.error("Error: One or both image files were not found. Please check the paths.")
except Exception as e:
    st.error(f"An error occurred while loading the images: {e}")

st.sidebar.success("Select a page from the navigation above.")

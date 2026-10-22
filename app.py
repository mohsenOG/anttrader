"""Streamlit scaffold for the future AntTrader research dashboard."""

import streamlit as st

st.set_page_config(page_title="AntTrader")
st.title("AntTrader")
st.write("Historical XAU/USD research on the 1-hour timeframe.")
st.info(
    "Topic 2 data loading and grid encoding are available through "
    "python -m anttrader.pipeline. Ant simulation, features, signals, "
    "backtesting, and the full dashboard remain pending."
)

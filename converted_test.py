import sentiment
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="ISD Threat Monitor",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# Sidebar
with st.sidebar:
    st.title('ISD Threat Monitor')

# Dashboard Main Panel
col = st.columns((4, 3, 2), gap='medium')

with col[0]: # word cloud
    st.subheader("Key Word Analysis")
    
    st.markdown('##### News')
    st.pyplot(sentiment.news_fig)

    st.markdown('##### Wikileaks')
    st.pyplot(sentiment.wikileaks_fig)


with col[1]: # sentiment pie charts
    #######################
    st.subheader("Sentiment Analysis")

    st.markdown('##### News')
    st.pyplot(sentiment.news_sentim_fig)

    st.markdown('##### Wikileaks')
    st.pyplot(sentiment.wikileaks_sentim_fig)
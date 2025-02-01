import sentiment
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from preprocessing import wikileaks_df, news_df

#######################
# Page Configuration
st.set_page_config(
    page_title="ISD Threat Monitor",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# Sidebar
with st.sidebar:
    st.title('ISD Threat Monitor')

# Sidebar Filters
st.sidebar.header("Filter Options")
sentiment_filter = st.sidebar.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])

# Modify existing sentiment pie chart values and colors dynamically
def update_pie_chart(fig, df):
    # Modify the existing pie chart in `sentiment.py`
    
    pos = (df['sentiment_score'] > 0).sum()
    neg = (df['sentiment_score'] < 0).sum()
    neu = (df['sentiment_score'] == 0).sum()

    # Define colors based on sentiment filter
    if sentiment_filter == "All":
        colors = ['green', 'red', 'blue']
    elif sentiment_filter == "Positive":
        colors = ['green', '#D3D3D3', '#D3D3D3']
    elif sentiment_filter == "Negative":
        colors = ['#D3D3D3', 'red', '#D3D3D3']
    elif sentiment_filter == "Neutral":
        colors = ['#D3D3D3', '#D3D3D3', '#576dff']

    # Update the existing figure
    ax = fig.axes[0]  # Access existing figure axes
    ax.clear()  # Clear existing pie chart

    # Redraw with updated values
    ax.pie([pos, neg, neu], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%', colors=colors)
    ax.set_title('Sentiment Analysis')

    return fig  # Return updated figure

# Update the existing figures in `sentiment.py`
sentiment.news_sentim_fig = update_pie_chart(sentiment.news_sentim_fig, news_df)
sentiment.wikileaks_sentim_fig = update_pie_chart(sentiment.wikileaks_sentim_fig, wikileaks_df)

#######################
# Display Dashboard
col = st.columns((4, 3, 2), gap='medium')

with col[0]:  # Word Cloud Section
    st.subheader("Key Word Analysis")

    st.markdown('##### News')
    st.pyplot(sentiment.news_fig)  # Word Cloud remains the same

    st.markdown('##### Wikileaks')
    st.pyplot(sentiment.wikileaks_fig)

with col[1]:  # Sentiment Pie Charts
    st.subheader("Sentiment Analysis")

    st.markdown('##### News')
    st.pyplot(sentiment.news_sentim_fig)  # Modified pie chart

    st.markdown('##### Wikileaks')
    st.pyplot(sentiment.wikileaks_sentim_fig)  # Modified pie chart

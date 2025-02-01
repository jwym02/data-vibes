import sentiment
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import entity
import topic

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
    st.title('⚠️ ISD Threat Monitor')

# Dashboard Main Panel
col = st.columns((3, 3, 5), gap='medium')

with col[0]: # word cloud
    st.subheader("Key Word Analysis")
    
    st.markdown('##### News')
    st.pyplot(sentiment.news_fig)

    st.markdown('##### Wikileaks')
    st.pyplot(sentiment.wikileaks_fig)


with col[1]: # sentiment pie charts
    st.subheader("Sentiment Analysis")

    st.markdown('##### News')
    st.pyplot(sentiment.news_sentim_fig)

    st.markdown('##### Wikileaks')
    st.pyplot(sentiment.wikileaks_sentim_fig)

with col[2]:
    st.subheader('##### Entity Relationship')
    st.pyplot(entity.entity_fig)

    

container = st.container()
with container:
    st.subheader("Topic Modeling")

    st.markdown('##### K Selection for LDA using Caojuan2009 and Deveaud2014')
    st.pyplot(topic.k_fig)
    
    st.markdown('##### Top 5 Words per Topic')
    st.dataframe(topic.topic_df)
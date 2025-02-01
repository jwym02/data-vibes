from preprocessing import wikileaks_df, news_df
import spacy
from itertools import combinations
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Function to extract relationships (co-occurrence of entities)
def extract_relationships(text):
    entities = extract_entities(text)
    return list(combinations(entities, 2)) if len(entities) > 1 else []

# Apply entity and relationship extraction to the dataset
wikileaks_df['Relationships'] = wikileaks_df['Text_No_Stopwords'].apply(extract_relationships)

from itertools import combinations
import pandas as pd

# Sample function to extract relationships based on co-occurrence
def extract_relationships(text, entities):
    # Find all entity pairs
    entity_pairs = list(combinations(entities, 2))
    return entity_pairs

# Example data for entities (mock entity extraction)
entities = ["airport", "administrative", "division"]

# Apply to the dataset
wikileaks_df['Relationships'] = wikileaks_df['Text_No_Stopwords'].apply(
    lambda x: extract_relationships(x, entities)  # Replace `entities` with actual entity extraction logic
)

def generate_word_cloud(df):
    all_text = " ".join(df['Text_No_Stopwords'])
    wordcloud = WordCloud(width=800, height=800, background_color='black').generate(all_text)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    return fig


# news_df word cloud
news_fig = generate_word_cloud(news_df)
# all_text = " ".join(news_df['Text_No_Stopwords'])
# wordcloud = WordCloud(width=800, height=800, background_color='grey').generate(all_text)
# news_fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(wordcloud, interpolation='bilinear')
# ax.axis('off')

# wikileaks_df word cloud
wikileaks_fig = generate_word_cloud(wikileaks_df)
# all_text = " ".join(wikileaks_df['Text_No_Stopwords'])
# wordcloud = WordCloud(width=800, height=800, background_color='gray').generate(all_text)
# wikileaks_fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(wordcloud, interpolation='bilinear')
# ax.axis('off')

##############
# news_df calculate polarity and subjectivity
def analyze_sentiment(text):
    blob = TextBlob(text)
    return pd.Series({'polarity': blob.sentiment.polarity,
                      'subjectivity': blob.sentiment.subjectivity})

news_df[['polarity', 'subjectivity']] = news_df['Text_No_Stopwords'].apply(analyze_sentiment)

# wikileaks_df calculate polarity and subjectivity
def analyze_sentiment(text):
    blob = TextBlob(text)
    return pd.Series({'polarity': blob.sentiment.polarity,
                      'subjectivity': blob.sentiment.subjectivity})

wikileaks_df[['polarity', 'subjectivity']] = wikileaks_df['Text_No_Stopwords'].apply(analyze_sentiment)


# news_df sentiment scores for each row
news_df['sentiment_score'] = news_df['Text_No_Stopwords'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Count the number of positive, negative, and neutral sentiments
pos = (news_df['sentiment_score'] > 0).sum()
neg = (news_df['sentiment_score'] < 0).sum()
neu = (news_df['sentiment_score'] == 0).sum()

# Generate and display a pie chart in Streamlit
# st.subheader("News Sentiment Analysis")
news_sentim_fig, news_sentim_ax = plt.subplots()
news_sentim_ax.pie(
    [pos, neg, neu],
    labels=['Positive', 'Negative', 'Neutral'],
    autopct='%1.1f%%',
    colors=['green', 'red', 'blue']
)
news_sentim_ax.set_title('Sentiment Analysis')


# wikileaks_df sentiment scores for each row
wikileaks_df['sentiment_score'] = wikileaks_df['Text_No_Stopwords'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Count the number of positive, negative, and neutral sentiments
pos = (wikileaks_df['sentiment_score'] > 0).sum()
neg = (wikileaks_df['sentiment_score'] < 0).sum()
neu = (wikileaks_df['sentiment_score'] == 0).sum()

# Generate and display a pie chart in Streamlit
# st.subheader("WikiLeaks Sentiment Analysis")
wikileaks_sentim_fig, wikileaks_sentim_ax = plt.subplots()
wikileaks_sentim_ax.pie(
    [pos, neg, neu],
    labels=['Positive', 'Negative', 'Neutral'],
    autopct='%1.1f%%',
    colors=['green', 'red', 'blue']
)
wikileaks_sentim_ax.set_title('Sentiment Analysis')

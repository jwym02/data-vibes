import spacy
from preprocessing import wikileaks_df, news_df, nlp
from itertools import combinations
from collections import Counter
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import numpy as np
import gensim
from gensim.models import LdaModel
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import hellinger
from tqdm import tqdm
import seaborn as sns

# Function to tokenize text using spaCy
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Apply spaCy tokenization
wikileaks_df['Tokenized_Text'] = wikileaks_df['Text_No_Stopwords'].apply(spacy_tokenizer)

# Create Dictionary
id2word = corpora.Dictionary(wikileaks_df['Tokenized_Text'])

# Create Corpus (Bag-of-Words representation)
corpus = [id2word.doc2bow(text) for text in wikileaks_df['Tokenized_Text']]

# Function to calculate CaoJuan2009 metric
def cao_juan_2009(dictionary, corpus, texts, min_k=2, max_k=15):
    scores = []
    for k in tqdm(range(min_k, max_k + 1)):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='u_mass')
        scores.append(cm.get_coherence())
    return scores

# Function to calculate Deveaud2014 metric
def deveaud_2014(dictionary, corpus, min_k=2, max_k=15):
    scores = []
    for k in tqdm(range(min_k, max_k + 1)):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        topics = lda.get_topics()
        pairwise_divergence = np.mean([hellinger(topics[i], topics[j])
                                       for i in range(len(topics)) for j in range(i+1, len(topics))])
        scores.append(pairwise_divergence)
    return scores

# Define range for K
min_k, max_k = 2, 15

# Calculate CaoJuan2009 and Deveaud2014 metrics
cao_juan_scores = cao_juan_2009(id2word, corpus, wikileaks_df['Tokenized_Text'], min_k, max_k)
deveaud_scores = deveaud_2014(id2word, corpus, min_k, max_k)

# Convert results to DataFrame for visualization
k_values = list(range(min_k, max_k + 1))
metrics_df = pd.DataFrame({"K": k_values, "CaoJuan2009": cao_juan_scores, "Deveaud2014": deveaud_scores})

# Create the figure and axis
k_fig, k_ax = plt.subplots(figsize=(10, 5))
k_ax.plot(metrics_df["K"], metrics_df["CaoJuan2009"], marker='o', label="CaoJuan2009 (Lower is better)")
k_ax.plot(metrics_df["K"], metrics_df["Deveaud2014"], marker='s', label="Deveaud2014 (Higher is better)")
k_ax.set_xlabel("Number of Topics (K)")
k_ax.set_ylabel("Score")
k_ax.set_title("K Selection for LDA using CaoJuan2009 and Deveaud2014")
k_ax.legend()
k_ax.grid(True)

# Define the number of topics
num_topics = 6

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=10, random_state=42)

# Function to get the dominant topic for a document
def get_dominant_topic(text):
    bow = id2word.doc2bow(text)
    topic_probs = lda_model.get_document_topics(bow)
    dominant_topic = max(topic_probs, key=lambda x: x[1])[0] if topic_probs else None
    return dominant_topic

# Apply to dataset
wikileaks_df['Dominant_Topic'] = wikileaks_df['Tokenized_Text'].apply(get_dominant_topic)

# Display the top 5 words for each topic
topics = lda_model.show_topics(num_topics=10, num_words=5, formatted=False)

# Convert topics to a structured table
topic_dict = {f"Topic {i+1}": [word[0] for word in topic[1]] for i, topic in enumerate(topics)}
topic_df = pd.DataFrame(topic_dict)

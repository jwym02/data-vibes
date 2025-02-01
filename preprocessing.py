import pandas as pd
import matplotlib as plt
import networkx as nx
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import hellinger
from tqdm import tqdm
import seaborn as sns

# Download necessary NLTK data files (if not already available)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Load the first Excel file
wikileaks_path = "./data/wikileaks_parsed.xlsx"
wikileaks_df = pd.read_excel(wikileaks_path)

# Load the second Excel file
news_path = "./data/news_excerpts_parsed.xlsx"
news_df = pd.read_excel(news_path)

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphanumeric characters and lowercase the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Apply preprocessing steps
wikileaks_df['Cleaned_Text'] = wikileaks_df['Text'].apply(preprocess_text)
news_df['Cleaned_Text'] = news_df['Text'].apply(preprocess_text)

# Remove duplicates and empty rows
wikileaks_df = wikileaks_df.drop_duplicates(subset=['Cleaned_Text']).dropna(subset=['Cleaned_Text'])
news_df = news_df.drop_duplicates(subset=['Cleaned_Text']).dropna(subset=['Cleaned_Text'])

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatization function
def lemmatize_text(text):
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

# Apply lemmatization to the cleaned text
wikileaks_df['Lemmatized_Text'] = wikileaks_df['Cleaned_Text'].apply(lemmatize_text)
news_df['Lemmatized_Text'] = news_df['Cleaned_Text'].apply(lemmatize_text)

# Replace '\n\n' and '\n' with a space
wikileaks_df['Cleaned_Text'] = wikileaks_df['Cleaned_Text'].str.replace(r'\n+', ' ', regex=True)
news_df['Cleaned_Text'] = news_df['Cleaned_Text'].str.replace(r'\n+', ' ', regex=True)

from nltk.corpus import stopwords
import nltk

# Download the stop words set
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Add custom stopwords to the existing stop words set
custom_stop_words = {
       "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "year", "years", "week", "weeks", "day", "days", "month", "months",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third",
    "united", "nations", "committee", "report", "review", "internal", "official", "officer", "program", "project",
    "airport", "terminal", "location", "place", "zone",
    "approximately", "next", "prior", "details", "percent", "inclusive",
    "summer", "winter", "fall", "spring",
    "memo", "dra", "control", "panel", "annual", "result",
    "us", "uk", "eu", "board", "group", "provisional",
    "department", "infrastructure", "agency", "staff", "heshe", "member", "mr", "however", "non", "g", "idoios"
}
stop_words.update(custom_stop_words)

# Function to remove stopwords (both default and custom)
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply the updated stop words removal to the cleaned text columns
wikileaks_df['Text_No_Stopwords'] = wikileaks_df['Cleaned_Text'].apply(remove_stopwords)
news_df['Text_No_Stopwords'] = news_df['Cleaned_Text'].apply(remove_stopwords)
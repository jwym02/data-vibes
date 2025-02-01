import spacy
from preprocessing import wikileaks_df, news_df
from itertools import combinations
from collections import Counter

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


# Function to extract named entities from cleaned text
def extract_entities_from_cleaned(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entity text and type
    return entities

# Apply entity extraction to the 'Text_No_Stopwords' column
wikileaks_df['Entities'] = wikileaks_df['Text_No_Stopwords'].apply(extract_entities_from_cleaned)

# Function to extract relationships based on entity co-occurrence
def extract_relationships(entities):
    # Create pairs of entities (combinations of 2)
    return list(combinations([entity[0] for entity in entities], 2))  # Use entity names only

# Apply the relationship extraction function
wikileaks_df['Relationships'] = wikileaks_df['Entities'].apply(extract_relationships)

# Flatten relationships into a list for visualization or analysis
all_relationships = [rel for sublist in wikileaks_df['Relationships'] for rel in sublist]

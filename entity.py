import spacy
from preprocessing import wikileaks_df, news_df, nlp
from itertools import combinations
from collections import Counter
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt

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

# Count relationship frequencies
relationship_counts = Counter(all_relationships)

# Convert to a DataFrame for easier analysis
relationship_df = pd.DataFrame(relationship_counts.items(), columns=["Entity Pair", "Count"])
relationship_df = relationship_df.sort_values(by="Count", ascending=False)

# Create a graph
G = nx.Graph()

# Add edges with weights (relationship counts)
for (entity1, entity2), count in relationship_counts.items():
    G.add_edge(entity1, entity2, weight=count)

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to extract and filter entities based on type
def extract_filtered_entities(text, allowed_types):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in allowed_types]
    return entities

# Allowed entity types (you can customize this list)
allowed_entity_types = {"PERSON", "ORG", "GPE"}  # Focus on people, organizations, and locations

# Apply entity extraction with filtering
wikileaks_df['Filtered_Entities'] = wikileaks_df['Text_No_Stopwords'].apply(
    lambda text: extract_filtered_entities(text, allowed_entity_types)
)

# Function to extract relationships from filtered entities
def extract_relationships(entities):
    return list(combinations([entity[0] for entity in entities], 2)) if len(entities) > 1 else []

# Apply relationship extraction to filtered entities
wikileaks_df['Filtered_Relationships'] = wikileaks_df['Filtered_Entities'].apply(extract_relationships)

# Flatten relationships into a list for analysis
all_filtered_relationships = [rel for sublist in wikileaks_df['Filtered_Relationships'] for rel in sublist]

# Count relationship frequencies
filtered_relationship_counts = Counter(all_filtered_relationships)

# Remove low-frequency relationships
min_frequency = 2
filtered_relationships = [
    (e1, e2) for (e1, e2), count in filtered_relationship_counts.items() if count >= min_frequency
]

# Create the graph
G_filtered = nx.Graph()
G_filtered.add_edges_from(filtered_relationships)

# Apply community detection for clustering
communities = community.greedy_modularity_communities(G_filtered)

# Assign colors to nodes by community
community_colors = {}
for i, com in enumerate(communities):
    for node in com:
        community_colors[node] = i

# Dynamically adjust node sizes, colors, and edge widths
node_sizes = [G_filtered.degree[node] * 100 for node in G_filtered.nodes()]
node_colors = [community_colors[node] for node in G_filtered.nodes()]
edge_widths = [filtered_relationship_counts.get((u, v), 1) for u, v in G_filtered.edges()]

# Extract edges, nodes, and weights from the graph
edges = list(G_filtered.edges(data=True))

# Create a DataFrame to tabulate the relationships
relationship_table = pd.DataFrame(
    [(edge[0], edge[1], edge[2].get("weight", 1)) for edge in edges],
    columns=["Entity 1", "Entity 2", "Weight"]
)

# Flatten the list of relationships and count occurrences
all_relationships = [rel for sublist in wikileaks_df['Filtered_Relationships'] for rel in sublist]
relationship_counts = Counter(all_relationships)  # Count each pair's frequency

# Create a graph and add edges with weights
G_weighted = nx.Graph()
for (entity1, entity2), weight in relationship_counts.items():
    G_weighted.add_edge(entity1, entity2, weight=weight)

# Extract edges with weights
edges_with_weights = list(G_weighted.edges(data=True))

# Create a DataFrame with weights
relationship_table = pd.DataFrame(
    [(edge[0], edge[1], edge[2]['weight']) for edge in edges_with_weights],
    columns=["Entity 1", "Entity 2", "Weight"]
)

# Adjust layout for better spacing
pos = nx.spring_layout(G_filtered, k=0.8)

# Draw the graph
entity_fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(
    G_filtered, pos, with_labels=False, node_size=node_sizes, node_color=node_colors,
    edge_color="gray", width=edge_widths, font_size=10, font_weight="bold", ax=ax
)

# Add labels with offset
for node, (x, y) in pos.items():
    ax.text(x, y + 0.02, s=node, bbox=dict(facecolor='white', alpha=0.5), fontsize=8, ha='center')

ax.set_title("Filtered and Clustered Entity Relationship Graph")
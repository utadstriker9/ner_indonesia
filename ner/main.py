import pandas as pd
import spacy
from IPython.display import HTML
import src.utils as utils

CONFIG_DATA=utils.config_load()

# Load the NER model
nlp = spacy.load('xx_ent_wiki_sm')

# Load your NER dataset (adjust the file path as needed)
ner_df = pd.read_csv(CONFIG_DATA['ner_data_path'], sep="\t")

# Create a dictionary mapping words to their corresponding tags
word_tag_dict = dict(zip(ner_df['word'], ner_df['tag']))

# Function to implement NER on the testing dataset
def implement_ner(text):
    doc = nlp(text)
    entities = [(token.text, word_tag_dict.get(token.text, "O")) for token in doc]
    return entities

# Load testing dataset 
test_df = utils.load_json(CONFIG_DATA['data_summarized_path'])

# Apply the NER function to create a new column with NER results
test_df['ner_results'] = test_df['summarized'].apply(implement_ner)

# Dictionary to map entity labels to fixed colors
entity_colors = {
    'O': None,  # No color for 'O' entities
    'B-ORGANIZATION': '#FFD700',  # Gold color for 'B-ORGANIZATION' entities
    'I-ORGANIZATION': '#FFD700',  # Gold color for 'I-ORGANIZATION' entities
    'L-ORGANIZATION': '#FFD700',  # Gold color for 'L-ORGANIZATION' entities
    'B-PERSON': '#00FF00',        # Green color for 'B-PERSON' entities
    'I-PERSON': '#00FF00',        # Green color for 'I-PERSON' entities
    'L-PERSON': '#00FF00',        # Green color for 'L-PERSON' entities
    'U-PERSON': '#00FF00',        # Green color for 'U-PERSON' entities
    'U-ORGANIZATION': '#FFD700',  # Gold color for 'U-ORGANIZATION' entities
    'B-TIME': '#FFA500',          # Orange color for 'B-TIME' entities
    'I-TIME': '#FFA500',          # Orange color for 'I-TIME' entities
    'L-TIME': '#FFA500',          # Orange color for 'L-TIME' entities
    'U-LOCATION': '#6495ED',      # Cornflower Blue color for 'U-LOCATION' entities
    'B-LOCATION': '#6495ED',      # Cornflower Blue color for 'B-LOCATION' entities
    'I-LOCATION': '#6495ED',      # Cornflower Blue color for 'I-LOCATION' entities
    'L-LOCATION': '#6495ED',      # Cornflower Blue color for 'L-LOCATION' entities
    'I-PERSON': '#00FF00',        # Green color for 'I-PERSON' entities
    'B-QUANTITY': '#9932CC',      # Dark Orchid color for 'B-QUANTITY' entities
    'L-QUANTITY': '#9932CC',      # Dark Orchid color for 'L-QUANTITY' entities
    'U-TIME': '#FFA500',          # Orange color for 'U-TIME' entities
    'I-QUANTITY': '#9932CC',      # Dark Orchid color for 'I-QUANTITY' entities
    'U-QUANTITY': '#9932CC'       # Dark Orchid color for 'U-QUANTITY' entities
}

# Function to visualize NER results as HTML with fixed colors for each entity label
def visualize_ner(text, entities):
    for ent_text, ent_label in entities:
        if isinstance(ent_label, str):
            ent_color = entity_colors.get(ent_label, None)
            if ent_color is not None:
                text = text.replace(ent_text, f'<span style="background-color: {ent_color};">{ent_text}</span>')
    return f'<p>{text}</p>'

# Display the DataFrame with NER results and visualization
pd.set_option('display.max_colwidth', None)
test_df['ner_visualization'] = test_df.apply(lambda row: visualize_ner(row['summarized'], row['ner_results']), axis=1)
HTML(test_df[['ner_visualization']].to_html(escape=False, header=False, index=False))
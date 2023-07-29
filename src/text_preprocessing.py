import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import utils

# Read Data
def read_data(return_file=False):
    data = pd.read_excel(CONFIG_DATA['raw_dataset_path'])

    # Print data
    print('ready read file, data shape   :', data.shape)

    # Dump data
    utils.dump_json(data, CONFIG_DATA['data_set_path'])

    # Return data
    if return_file:
        return data
    
# Cleaning Data
def clean_data(data, return_file=True):
    # Remove multiple used of '='
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].dropna()
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace('=', ' ')
    
    # Remove '\n', '\r', and '\t'
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace('\n', ' ')
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace('\r', ' ')
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace('\t', ' ')
    
    # Remove Non ASCII
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].str.encode('ascii', 'ignore').str.decode('ascii')

    # Remove Multiple Space
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].replace(r'\s+', ' ', regex=True)
    
    # Remove Whitespace in Start and End
    data[CONFIG_DATA['text_column']] = data[CONFIG_DATA['text_column']].str.strip()
    
    # Tokenizing 
    data['tokens'] = data[CONFIG_DATA['text_column']].replace(r'[^0-9a-zA-Z ]', '', regex=True).replace(r'\s+', ' ', regex=True).astype(str).apply(word_tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: [len(token) for token in x]).apply(sum)
    
    if return_file:
        return data

# Generate Preprocessor
def generate_preprocessor(return_file=False):
    # Load Data
    data = utils.load_json(CONFIG_DATA['data_set_path'])
    data = clean_data(data)
    
    # Print Data
    print('ready processed, data shape   :', data.shape)
    
    # Dump Data
    utils.dump_json(data, CONFIG_DATA['data_clean_path'])    
    
    if return_file:
        return data
    
if __name__ == '__main__':
    # 1. Load configuration file
    CONFIG_DATA = utils.config_load()
    
    # 2. Read Data
    read_data()

    # 3. Generate Preprocessor
    generate_preprocessor()

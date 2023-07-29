import utils
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import pandas as pd
import numpy as np

CONFIG_DATA = utils.config_load()

# Summarizing
def generate_summarization(data):
    input_ids = tokenizer(data, return_tensors='pt', padding=True).to(device)
    input_token = input_ids["input_ids"]
    
    with torch.no_grad():
        summary_ids = model.generate(
            input_token,
            min_length=CONFIG_DATA['min_length'],
            max_length=CONFIG_DATA['max_length'],
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True
        )

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary_text

# Transform to Array Data
def load_data_to_array(key=[CONFIG_DATA['text_column'], CONFIG_DATA['sum_token_column']]):
    
    # Read Data
    data = utils.load_json(CONFIG_DATA['data_clean_path'])
    
    data_key = data[key].reset_index()
    arr_i_key = data_key[['index'] + key].values
    
    return arr_i_key

# Run Summarizator
def run_summarization(arr):
    arr_generated = []
    
    i = 0
    for row in tqdm(arr):
        index = row[0]
        text = row[1]
        try:
            sg = generate_summarization(text)
            generated = [[index, sg]]
            arr_generated = [*arr_generated, *generated]
        except Exception as e:
            errors = [*errors, *[index]]
                
    print(f'Generated {len(arr_generated)}, Errors {len(errors)}')
    
    return np.array(arr_generated)

# Save Summarized Data
def save_summarization(sum_arr):
    df_gen = pd.DataFrame(sum_arr)
    df_gen = df_gen.rename(columns={
        0: 'index',
        1: 'summarized'
    })
    df_gen['index'] = df_gen['index'].astype(int)
    df_gen = df_gen[df_gen['index'] != -1]
    
    # Load Data
    data = utils.load_json(CONFIG_DATA['data_clean_path']).reset_index()
    
    # Merge Data
    df_merge = data.merge(df_gen, left_on='index', right_on='index', how='left')
    data = df_merge.set_index('index')
    
    # Dump Data
    utils.dump_json(data, CONFIG_DATA['data_summarized_path'])
    
    return df_merge
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model = T5ForConditionalGeneration.from_pretrained(CONFIG_DATA['sum_pretrained'])
    model = t5_model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(CONFIG_DATA['sum_pretrained'])
    
    # 2. Generate Summarizator
    arr = load_data_to_array()
    sum_arr = run_summarization(arr)
    save_summarization(sum_arr)
    
import pandas as pd
import glob
import re
from tqdm import tqdm

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split())
    return text

def create_df_convote(filenames):
    items = []
    for file_path in tqdm(filenames):
        fp = open(file_path, encoding='utf8')
        text = fp.read()
        fp.close()
        text = clean_text(text)

        # get metadata
        filename = file_path.split('\\')
        filename_tokens = filename[-1].split('_')
        bill_debate_number = filename_tokens[0]

        party = filename_tokens[-1][0] # D or R
        if party == 'R':
            political_party = 1
        else:
            political_party = 0

        m = filename_tokens[-1][1]
        if m == 'M':
            mentioned_bill = 1
        else:
            mentioned_bill = 0

        vote = filename_tokens[-1][2]
        if vote == 'Y':
            voted_yes = 1
        else:
            voted_yes = 0
        # add to list
        items.append([text, bill_debate_number, political_party, mentioned_bill, voted_yes])
    
    # convert list to df
    new_df = pd.DataFrame(items)
    return new_df.rename(index=str, columns={0: 'sentence', 1: 'bill_debate_number', 2: 'labels', 3: 'mentioned_bill', 4: 'voted_yes'})

# main
filenames_train = glob.glob('data/convote_v1.1/data_stage_one/training_set/*.txt')
filenames_test = glob.glob('data/convote_v1.1/data_stage_one/test_set/*.txt')

df_train = create_df_convote(filenames_train)
df_test = create_df_convote(filenames_test)

df_train.to_csv('df_convote_train.csv')
df_test.to_csv('df_convote_test.csv')
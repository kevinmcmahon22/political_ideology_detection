import pandas as pd
import glob
from tqdm import tqdm

filenames_train = glob.glob('data/convote_v1.1/data_stage_one/training_set/*.txt')
filenames_test = glob.glob('data/convote_v1.1/data_stage_one/test_set/*.txt')

def create_df(filenames):
    items = []
    for file in tqdm(filenames):
        fp = open(file, encoding='utf8')
        text = fp.read()
        fp.close()

        filename_tokens = file.split('_')
        bill_debate_number = filename_tokens[0]
        party = filename_tokens[-1][0] # D or R
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

        items.append([text, bill_debate_number, party, mentioned_bill, voted_yes])

    new_df = pd.DataFrame(items)
    return new_df.rename(index=str, columns={0: 'text', 1: 'bill_debate_number', 2: 'party', 3: 'mentioned_bill', 4: 'voted_yes'})

df_train = create_df(filenames_train)
df_test = create_df(filenames_test)

df_train.to_csv('df_train')
df_test.to_csv('df_test')
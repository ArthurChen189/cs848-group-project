import pandas as pd

parent_table = pd.read_csv('./raw/store.csv')
train_table = pd.read_csv('./raw/train.csv')
test_table = pd.read_csv('./raw/test.csv')

# we merge train and test tables to get all the data
all_table = pd.concat([train_table, test_table])

# we only keep data after 2015-06-01
all_table = all_table[all_table['Date'] >= '2015-06-01']

print(f"There are {len(all_table)} rows in the merged table")

# export
all_table.to_csv('./preprocessed/all_table.csv', index=False)
parent_table.to_csv('./preprocessed/parent_table.csv', index=False)

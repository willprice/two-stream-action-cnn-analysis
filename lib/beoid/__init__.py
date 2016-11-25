import pandas as pd

label_to_id = pd.read_csv('/home/will/data/beoid/beoid-class-labels.csv')
label_count = len(label_to_id)


def get_label_id(id):
    row = label_to_id.loc[id]
    return row.label

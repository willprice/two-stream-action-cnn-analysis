import pandas as pd

label_to_id = pd.read_csv('/home/will/thesis/data/beoid/beoid-class-labels.csv')
label_count = len(label_to_id)


def get_label_id(label):
    row = label_to_id.loc[label_to_id.label == label]
    return row.id

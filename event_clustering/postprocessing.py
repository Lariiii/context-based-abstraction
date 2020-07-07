import pandas as pd
from itertools import product
from string import ascii_uppercase

def add_cluster_label(df, df_vectorized, cluster_alg):
    new_col = df_vectorized._get_numeric_data().dropna(axis=1)
    df["cluster_label"] = pd.Series(cluster_alg.predict(new_col), index=df.index)
    return df

def replace_with_representative(df, col_name, col_label, original_df_columns):
    representative_dict = dict()
    cluster_nr = len(df[col_label].unique())
    alphabet_len = len(ascii_uppercase)
    repetitions = (float(cluster_nr) / float(alphabet_len)) + 1
    representatives = [''.join(i) for i in product(ascii_uppercase, repeat = int(repetitions))]
    for label in range(cluster_nr):
        representative_dict[label] = representatives[label]
        
    df[col_name] = df[col_label].map(representative_dict)
    # just keep original columns
    df = df[original_df_columns]
    return df

def write_to_csv(df, filename, index=False):
    df.to_csv(filename, index=index)
    print("Finished writing to CSV file.")


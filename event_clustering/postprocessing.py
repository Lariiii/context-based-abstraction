import pandas as pd
from itertools import product
from string import ascii_uppercase

def add_cluster_label(df, df_vectorized, cluster_alg):
    new_col = df_vectorized._get_numeric_data().dropna(axis=1)
    df["cluster_label"] = pd.Series(cluster_alg.predict(new_col), index=df.index)
    return df

def replace_with_representative(df, col_name, cluster_col_name):
    representative_dict = dict()
    amount_of_clusters = len(df[cluster_col_name].unique())
    alphabet_len = len(ascii_uppercase)
    repetitions = (float(amount_of_clusters) / float(alphabet_len)) + 1
    representatives = [''.join(i) for i in product(ascii_uppercase, repeat = int(repetitions))]
    for label in range(amount_of_clusters):
        representative_dict[label] = representatives[label]
        
    df[col_name] = df[cluster_col_name].map(representative_dict)
    return df

def write_to_csv(df, filename, index=False):
    df.to_csv(filename, index=index)
    print("Finished writing to CSV file.")


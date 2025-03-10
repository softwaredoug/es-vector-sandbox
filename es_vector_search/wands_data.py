import pandas as pd
import numpy as np
import tarfile
import os


UNZIPPED_DIR = os.path.expanduser('~/.wands')

os.makedirs(UNZIPPED_DIR, exist_ok=True)

PRODUCTS_FILE = os.path.join(UNZIPPED_DIR, 'dataset/product.csv')
QUERIES_FILE = os.path.join(UNZIPPED_DIR, 'dataset/query.csv')
LABELS_FILE = os.path.join(UNZIPPED_DIR, 'dataset/label.csv')


def unzip_wands_dataset():
    """Untar/unzip wands dataset ta data/WANDS/dataset.tar.gz."""
    if os.path.exists(PRODUCTS_FILE):
        return
    path = 'data/WANDS/dataset.tar.gz'
    with tarfile.open(path, 'r:gz') as tar:
        tar.extractall(UNZIPPED_DIR)


def _wands_data_merged():
    unzip_wands_dataset()
    products = pd.read_csv(PRODUCTS_FILE, delimiter='\t')
    queries = pd.read_csv(QUERIES_FILE, delimiter='\t')
    labels = pd.read_csv(LABELS_FILE, delimiter='\t')
    labels.loc[labels['label'] == 'Exact', 'grade'] = 2
    labels.loc[labels['label'] == 'Partial', 'grade'] = 1
    labels.loc[labels['label'] == 'Irrelevant', 'grade'] = 0
    labels = labels.merge(queries, how='left', on='query_id')
    labels = labels.merge(products, how='left', on='product_id')
    return labels


def wands_products():
    unzip_wands_dataset()
    return pd.read_csv(PRODUCTS_FILE, delimiter='\t')


def pairwise_df(n, seed=42, filter_same_label=False):
    labels = _wands_data_merged()

    # Sample n rows
    labels = labels.sample(10000, random_state=seed)

    # Get pairwise
    pairwise = labels.merge(labels, on='query_id')
    # Shuffle completely, otherwise they're somewhat sorted on query
    pairwise = pairwise.sample(frac=1, random_state=seed)

    # Drop same id
    pairwise = pairwise[pairwise['product_id_x'] != pairwise['product_id_y']]

    # Drop same rating
    if filter_same_label:
        pairwise = pairwise[pairwise['label_x'] != pairwise['label_y']]

    assert n <= len(pairwise), f"Only {len(pairwise)} rows available"
    return pairwise.head(n)


def queries_sample(num_queries=100, num_docs=10, seed=420):
    np.random.seed(seed)
    labels = _wands_data_merged()
    queries = labels['query'].unique()
    queries = np.random.choice(queries, num_queries, replace=False)
    docs_per_query = labels[labels['query'].isin(queries)]
    # Shuffle randomly
    docs_per_query = docs_per_query.sample(frac=1, random_state=seed)
    docs_per_query = labels.groupby('query').head(num_docs).reset_index(drop=True)
    return docs_per_query

import pandas as pd
import numpy as np
import tarfile
import os
from torch.utils.data import Dataset


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
    products['product_name'].fillna('', inplace=True)
    products['product_description'].fillna('', inplace=True)
    return labels


def eval_results(results: pd.DataFrame, at=10):
    """Results are all results every search and will be joined with the labels. and NDCGG computed."""
    labels = wands_labels()
    results = results.merge(labels, on=['query_id', 'product_id'], how='left')
    results = results.sort_values(['query_id', 'rank'])
    results['dcg'] = results['grade'] / np.log2(results['rank'] + 1)
    # Compute IDCG from labels
    labels = labels.sort_values(['query_id', 'grade'], ascending=False)
    labels['rank'] = labels.groupby('query_id').cumcount() + 1
    labels = labels[labels['rank'] <= at]
    labels['idcg'] = labels['grade'] / np.log2(labels['rank'] + 1)
    results = results[results['rank'] <= at]
    ndcgs = results.groupby(['query_id', 'query'])['dcg'].sum() / labels.groupby('query_id')['idcg'].sum()
    return ndcgs


def wands_labels():
    unzip_wands_dataset()
    labels = pd.read_csv(LABELS_FILE, delimiter='\t')
    labels.loc[labels['label'] == 'Exact', 'grade'] = 2
    labels.loc[labels['label'] == 'Partial', 'grade'] = 1
    labels.loc[labels['label'] == 'Irrelevant', 'grade'] = 0
    return labels.groupby(['query_id', 'product_id']).head(1).reset_index(drop=True)


def wands_products():
    unzip_wands_dataset()
    return pd.read_csv(PRODUCTS_FILE, delimiter='\t')


def wands_queries():
    unzip_wands_dataset()
    return pd.read_csv(QUERIES_FILE, delimiter='\t')


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


class WANDSDataset(Dataset):
    """
    PyTorch Dataset for the WANDS product search dataset.
    Produces (query, product_name, product_description, grade).
    """

    def __init__(self):
        self.data = _wands_data_merged().reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        query = row['query']
        product_name = row['product_name'] if not isinstance(row['product_name'], float) else ''
        product_description = row['product_description'] if not isinstance(row['product_description'], float) else ''
        grade = int(row['grade'])
        return query, product_name, product_description, grade

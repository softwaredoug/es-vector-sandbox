from es_vector_search.wands_data import wands_products
from es_vector_search.log_stdout import log_to_stdout
from es_vector_search.embedder import TextEmbedder
import pandas as pd
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import logging
from tqdm import tqdm


log_to_stdout("es_vector_search")
log_to_stdout(__name__)


logger = logging.getLogger(__name__)


def wands_products_to_bulk(wands: pd.DataFrame):
    count = len(wands)
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    name_embeddings = embedder(wands['product_name'].values)
    import pdb; pdb.set_trace()
    for idx, row in tqdm(wands.iterrows(), total=count, desc="Indexing WANDS products"):
        yield {
            "_index": "wands_products",
            "_id": row['product_id'],
            "_source": {
                "product_name_minilm": name_embeddings[idx].tolist(),
                "product_name": row['product_name'] if pd.notna(row['product_name']) else "",
                "product_description": row['product_description'] if pd.notna(row['product_description']) else "",
                "product_category": row['category hierarchy'] if pd.notna(row['category hierarchy']) else "",
                "product_class": row['product_class'] if pd.notna(row['product_class']) else "",
            }
        }


def index(es_client: Elasticsearch, docs, index_name: str = "wands"):
    settings = open("es_vector_search/settings.json").read()
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
    es_client.indices.create(index=index_name, body=settings)
    bulk(es_client, docs)


def main():
    es_client = Elasticsearch("http://localhost:9200")
    wands = wands_products()
    docs = wands_products_to_bulk(wands)
    index(es_client, docs)


if __name__ == "__main__":
    main()

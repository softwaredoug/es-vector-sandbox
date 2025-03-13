"""Search with argv1."""
from elasticsearch import Elasticsearch
from es_vector_search.embedder import TextEmbedder
from sys import argv
import pandas as pd
from es_vector_search.wands_data import wands_products, wands_queries, eval_results
from time import perf_counter


from es_vector_search.stored_cache import StoredLRUCache
import random


@StoredLRUCache(maxsize=1000)
def minilm(query: str):
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedder([query])[0]


def search_baseline(es: Elasticsearch, query: str):
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["product_name", "product_description", "product_class"]
            }
        }
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def search_knn(es: Elasticsearch, query: str):
    query_vector = minilm(query)
    body = {
        "query": {
            "knn": {
                "field": "product_name_description_minilm",
                "query_vector": query_vector.tolist(),
            }
        },
        "_source": ["product_name", "product_description"]
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def search_hybrid(es: Elasticsearch, query: str):
    query_vector = minilm(query)
    body = {
        "query": {
            "knn": {
                "field": "product_name_description_minilm",
                "query_vector": query_vector.tolist(),
                "filter": {
                    "multi_match": {
                        "query": query,
                        "fields": ["product_name", "product_description", "product_class"]
                    }
                }
            }
        },
        "_source": ["product_name", "product_description"]
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def search_hybrid_dismax(es: Elasticsearch, query: str):
    query_vector = minilm(query)
    body = {
        "query": {
            "dis_max": {
                "queries": [
                    # Hybrid clause
                    {"knn": {
                        "field": "product_name_description_minilm",
                        "query_vector": query_vector.tolist(),
                        "filter": {
                            "multi_match": {
                                "query": query,
                                "fields": ["product_name", "product_description", "product_class"]
                            }
                        },
                        "boost": 10.0
                    }},
                    # Fallback
                    {"knn": {
                        "field": "product_name_description_minilm",
                        "query_vector": query_vector.tolist(),
                        "boost": 0.1
                    }}
                ]
            }
        },
        "_source": ["product_name", "product_description"]
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def search_hybrid_dismax_name_boosted(es: Elasticsearch, query: str):
    query_vector = minilm(query)
    body = {
        "query": {
            "bool": {
                "should": [
                    # A title boost
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["product_name"],
                            "boost": 0.1
                        }
                    },
                ],
                "must": [
                    {"dis_max": {
                        "queries": [
                            # Hybrid clause
                            {"knn": {
                                "field": "product_name_description_minilm",
                                "query_vector": query_vector.tolist(),
                                "filter": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["product_name", "product_description", "product_class"]
                                    }
                                },
                                "boost": 10.0
                            }},
                            # Fallback
                            {"knn": {
                                "field": "product_name_description_minilm",
                                "query_vector": query_vector.tolist(),
                                "boost": 0.1
                            }}
                        ]
                    }}
                ]
            }
        },
        "_source": ["product_name", "product_description"]
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def search_hybrid_dismax_boosted_floor(es: Elasticsearch, query: str):
    query_vector = minilm(query)
    body = {
        "query": {
            "bool": {
                "should": [
                    # A title boost
                    {"multi_match": {
                            "query": query,
                            "fields": ["product_name"],
                            "boost": 0.1
                        }
                    },
                    # A name vector boost
                    {"knn": {
                        "query_vector": query_vector.tolist(),
                        "field": "product_name_minilm",
                        "boost": 0.1
                    }}
                ],
                "must": [
                    {"dis_max": {
                        "queries": [
                            # Hybrid clause
                            {"knn": {
                                "field": "product_name_description_minilm",
                                "query_vector": query_vector.tolist(),
                                "filter": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["product_name", "product_description", "product_class"]
                                    }
                                },
                                "boost": 10.0,
                            }},
                            # Fallback
                            {"knn": {
                                "field": "product_name_description_minilm",
                                "query_vector": query_vector.tolist(),
                                "boost": 0.1,
                            }},
                        ]
                    }}
                ]
            }
        },
        "_source": ["product_name", "product_description"]
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def rrf(es: Elasticsearch, query: str, search_fn1=search_knn, search_fn2=search_baseline):
    """Implement reciprocal rank fusion using search_fn1 and search_fn2."""
    hits1 = search_fn1(es, query)
    df : list | pd.DataFrame = []
    for idx, hit in enumerate(hits1['hits']['hits']):
        df.append({
            "product_id": hit['_id'],
            "product_name": hit['_source']['product_name'],
            "product_description": hit['_source']['product_description'],
            "score": hit['_score'],
            "rank": idx + 1,
            "reciprocal_rank": 1 / (idx + 1)
        })
    hits2 = search_fn2(es, query)
    for idx, hit in enumerate(hits2['hits']['hits']):
        df.append({
            "product_id": hit['_id'],
            "product_name": hit['_source']['product_name'],
            "product_description": hit['_source']['product_description'],
            "score": hit['_score'],
            "rank": idx + 1,
            "reciprocal_rank": 1 / (idx + 1)
        })
    df = pd.DataFrame(df)
    df = df.groupby('product_id').agg({
        "product_name": "first",
        "product_description": "first",
        "score": "mean",
        "rank": "mean",
        "reciprocal_rank": "sum"
    })
    df = df.sort_values("reciprocal_rank", ascending=False)
    # Back to hits
    hits = []
    for idx, row in df.iterrows():
        hits.append({
            "_id": idx,
            "_score": row['reciprocal_rank'],
            "_source": {
                "product_name": row['product_name'],
                "product_description": row['product_description']
            }
        })
    return {"hits": {"hits": hits}}



def search_all(es: Elasticsearch, search_fn=search_knn, at=10):
    queries = wands_queries()
    results = []
    for query_idx, query in queries.iterrows():
        hits = search_fn(es, query['query'])
        for idx, hit in enumerate(hits['hits']['hits']):
            results.append({
                "query_id": query['query_id'],
                "query": query['query'],
                "product_id": int(hit['_id']),
                "product_name": hit['_source']['product_name'],
                "rank": idx + 1,
                "score": hit['_score']
            })
            if idx >= at:
                break
    results = pd.DataFrame(results)
    ndcgs = eval_results(results)
    median_ndcg = ndcgs.median()
    mean_ndcg = ndcgs.mean()
    print(f"Mean NDCG: {mean_ndcg}")
    print(f"Median NDCG: {median_ndcg}")


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")
    if len(argv) < 2:
        for fn in [search_baseline, search_knn, rrf, search_hybrid,
                   search_hybrid_dismax, search_hybrid_dismax_name_boosted,
                   search_hybrid_dismax_boosted_floor]:
            fn_name = fn.__name__.replace("search_", "")
            print(f"Running {fn_name}")
            start = perf_counter()
            search_all(es, search_fn=fn)
    elif argv[1] == "baseline":
        search_all(es, search_fn=search_baseline)
    elif argv[1] == "knn":
        search_all(es, search_fn=search_knn)
    elif argv[1] == "hybrid":
        search_all(es, search_fn=search_hybrid)
    elif argv[1] == "hybrid_dismax":
        search_all(es, search_fn=search_hybrid_dismax)
    elif argv[1] == "hybrid_dismax_name_boosted":
        search_all(es, search_fn=search_hybrid_dismax_name_boosted)
    elif argv[1] == "hybrid_dismax_boosted_floor":
        search_all(es, search_fn=search_hybrid_dismax_boosted_floor)
    elif argv[1] == "rrf":
        search_all(es, search_fn=lambda es, query: rrf(es, query, search_fn1=search_knn, search_fn2=search_baseline))

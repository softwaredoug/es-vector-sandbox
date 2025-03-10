"""Search with argv1."""
from elasticsearch import Elasticsearch
from es_vector_search.embedder import TextEmbedder
from sys import argv


def search(es: Elasticsearch, query: str):
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_vector = embedder([query])[0]
    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": """
                        double sim = cosineSimilarity(params.query_vector, 'product_name_minilm');
                        if (sim < params.threshold) {
                            return 0.0;
                        }
                        double max = Math.pow(2, params.rate) - 1.0;
                        return (Math.pow(2, (sim * params.rate) - params.threshold) - 1.0) / max;
                    """,
                    "params": {
                        "query_vector": query_vector.tolist(),
                        "threshold": 0.2,
                        "rate": 10.0
                    }
                },
            }
        },
        "min_score": 0.1,
        "size": 1000
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def main():
    query = argv[1]
    es = Elasticsearch("http://localhost:9200")
    hits = search(es, query)
    for idx, hit in enumerate(hits['hits']['hits']):
        score = hit['_score']
        print(idx, score, hit['_source']['product_name'])


if __name__ == "__main__":
    main()

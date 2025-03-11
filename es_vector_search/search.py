"""Search with argv1."""
from elasticsearch import Elasticsearch
from es_vector_search.embedder import TextEmbedder
from sys import argv
import pandas as pd
from es_vector_search.wands_data import wands_products

import random


def unique_product_classes():
    products = wands_products()
    return products['product_class'].unique()


def search(es: Elasticsearch, query: str, method: int,
           threshold, rate, delta):
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"Query: {query}")
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
                        if (params.method == 2) {
                            // logistic
                            return 1.0 / (1.0 + Math.exp(-params.rate * (sim - params.delta)));
                        }
                        return sim;
                    """,
                    "params": {
                        "query_vector": query_vector.tolist(),
                        "threshold": threshold,
                        "rate": rate,
                        "delta": delta,
                        "method": int(method)
                    }
                },
            }
        },
        "min_score": threshold,
        "size": 4000
    }
    hits = es.search(index="wands_products", body=body)
    return hits


def run_for_query(es: Elasticsearch, search_query: str,
                  method: int,
                  num_with_class: int,
                  rate=1.0, threshold=0.5, delta=0.5, min_score=0.05):
    query_scores = []
    hits = search(es, search_query, method,
                  rate=rate, threshold=threshold, delta=delta, min_score=min_score)
    first_score = None
    num_hits_with_expected_class = 0
    num_hits = len(hits['hits']['hits'])
    print(f"Method: {method}, hits: {num_hits}")
    for idx, hit in enumerate(hits['hits']['hits']):
        score = hit['_score']
        if len(query_scores) <= idx:
            query_scores.append({"query": search_query,
                                 "product_name": hit['_source']['product_name'],
                                 "product_class": hit['_source']['product_class']})
        if hit['_source']['product_class'] == search_query:
            num_hits_with_expected_class += 1
        query_scores[idx][method] = score
        if first_score is None:
            first_score = score
    if num_hits == 0:
        prec = 0
        recall = 0
    else:
        prec = num_hits_with_expected_class / num_hits
        recall = num_hits_with_expected_class / num_with_class
    if prec == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    for row in query_scores:
        row["prec"] = prec
        row["recall"] = recall
        row["f1"] = f1
    print(f"Method: {method}, prec: {prec}, recall: {recall}, f1: {f1}")
    return query_scores

#
# Best f1: 3.41765445394654
# Best parameters: rate: 11.70648369010722, threshold: 0.5575076215737774, delta: 0.312153077
# 47103216, min_score: 0.42392917657658
#
#
item_type_classes = ['Beds', 'Vanities', 'Sofas', 'Bookcases', 'Desks', 'Ottomans', 'Cat Beds', 'Pergolas',
                     'Area Rugs', 'Planters', 'Recliners', 'Wreaths', 'Fountains', 'Candles', 'Mailboxes',
                     'Safes', 'Doormats', 'Benches', 'Dollies', 'Wall Art', 'Trunks', 'Bean Bags', 'Futons',
                     'Awnings', 'Saunas', 'Sheds', 'TV Trays', 'Pet Gates', 'Cabinetry', 'Hot Tubs', 'Bar Carts',
                     'Valves', 'Gliders', 'Cooktops', 'Dryers', 'Bookends', 'Woks', 'Cake Pans', 'Hangers',
                     'Chairmats', 'Hammocks', 'Fabric', 'Arbors', 'Coolers', 'Flags', 'TV Mounts', 'Cribs',
                     'Slides', 'Mobiles', 'Grab Bars', 'Wallpaper', 'Mantels', 'Lockers', 'Pot Racks', 'Deadbolts',
                     'Wedding', 'Drains', 'Beer Pong', 'Rug Pads', 'Trellises', 'Steam', 'Ranges', 'Ramps',
                     'Hooks', 'Teen Beds', 'Globes', 'Bidets']
#
# Best -- f1 21.95914490677431 rate: 10.337777461179622, threshold: 0.455822073094613, delta:
# 0.4806703818358925, min_score: 0.44582550638546864
#Best f1: 21.95914490677431
#Best parameters: rate: 10.337777461179622, threshold: 0.455822073094613, delta: 0.480670381
# 8358925, min_score: 0.44582550638546864


def benchmark_classes(method: int):
    es = Elasticsearch("http://localhost:9200")
    all_queries = []
    products = wands_products()
    classes = item_type_classes
    print(f"Processing {len(classes)} classes")
    attempts  = 50
    max_sum_f1 = 0.0
    best_rate = 0.0
    best_threshold = 0.0
    best_delta = 0.0
    best_min_score = 0.0
    for attempt in range(attempts):
        rate = random.uniform(0.1, 20)
        threshold = random.uniform(0.1, 0.9)
        delta = random.uniform(0.1, 0.9)
        min_score = random.uniform(0.05, 0.9)

        sum_prec = 0
        sum_recall = 0
        sum_f1 = 0

        for class_no, classs in enumerate(classes):
            if isinstance(classs, float):
                continue
            num_with_class = len(products[products['product_class'] == classs])
            query_scores = []
            results = run_for_query(es, classs, method, num_with_class,
                                    rate=rate, threshold=threshold, delta=delta, min_score=min_score)
            query_scores.extend(results)
            sum_prec += results[0]['prec'] if len(results) > 0 else 0
            sum_recall += results[0]['recall'] if len(results) > 0 else 0
            sum_f1 += results[0]['f1'] if len(results) > 0 else 0
            avg_prec = sum_prec / (class_no + 1)
            avg_recall = sum_recall / (class_no + 1)
            avg_f1 = sum_f1 / (class_no + 1)
            print(f"Query {classs} ***")
            print(f"Parameters: rate: {rate}, threshold: {threshold}, delta: {delta}, min_score: {min_score}")
            print(f"Method {method} Average prec: {avg_prec}, average recall: {avg_recall}, average f1: {avg_f1}")
            print(f"Best -- f1 {max_sum_f1} rate: {best_rate}, threshold: {best_threshold}, delta: {best_delta}, min_score: {best_min_score}")
            all_queries.extend(query_scores)
        if sum_f1 > max_sum_f1:
            max_sum_f1 = sum_f1
            best_rate = rate
            best_threshold = threshold
            best_delta = delta
            best_min_score = min_score
            print(f"New best f1: {max_sum_f1}")
            print(f"New best parameters: rate: {best_rate}, threshold: {best_threshold}, delta: {best_delta}, min_score: {best_min_score}")
    print(f"Best f1: {max_sum_f1}")
    print(f"Best parameters: rate: {best_rate}, threshold: {best_threshold}, delta: {best_delta}, min_score: {best_min_score}")


if __name__ == "__main__":
    query = argv[1]
    es = Elasticsearch("http://localhost:9200")
    hits_orig = search(es=es, query=query,
                       method=0, threshold=0.08, rate=10.0, delta=0.8)
    hits = search(es=es, query=query,
                  method=2, threshold=0.08, rate=10.0, delta=0.8)
    for idx, (hit_orig, hit_logistic) in enumerate(zip(hits_orig['hits']['hits'], hits['hits']['hits'])):
        assert hit_orig['_source']['product_name'] == hit_logistic['_source']['product_name']
        print(idx, hit_orig['_source']['product_name'],
              hit_orig['_score'], hit_logistic['_score'])

#    if argv[1] == "benchmark":
#        benchmark_classes(2)
#        exit(0)
#    else:
#        products = wands_products()
#        classes = item_type_classes
#        search_class = classes[int(argv[1])]
#        num_with_class = len(products[products['product_class'] == search_class])
#        method = 2
#        results = run_for_query(Elasticsearch("http://localhost:9200"), search_class,
#                                method=method,
#                                num_with_class=num_with_class)
#        if len(results) == 0:
#            print(f"No results for query {search_class}")
#            exit(0)
#        print(pd.DataFrame(results)[[method, 'query', 'product_name', 'product_class']].to_markdown())
#        prec = results[0]['prec']
#        recall = results[0]['recall']
#        f1 = results[0]['f1']
#        print(f"Quiery {search_class} - Method: {method} - prec: {prec}, recall: {recall}, f1: {f1}")

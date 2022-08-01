"""This script is for retrieving the top 1000 documents based on the queries"""

# importing libs
import requests

import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()


TOP_K = 1000
INDEX_NAME = "ms_marco"
HOSTNAME = "localhost"
PORT = 9200
PASSWORD = "set your elastic password"
RUN_NAME = "ELASTIC"
OUTPUT_FILE_NAME = ""
ERROR_FILE_NAME = ""

# validating the number of documents
res = requests.get(
    f"http://elastic:{PASSWORD}@{HOSTNAME}:{PORT}/{INDEX_NAME}/_count").json()
assert res['count'] == 11959635


# preparing the queries
train_queries_path = "/home/ali_najafi/docv2_train_queries.tsv"

train_df = pd.read_csv(train_queries_path, sep="\t", header=None).rename(
    columns={0: "q_id", 1: "q_text"})
print("train queries are loaded!")


def query_gen(q_id, q_text):
    q_text = re.sub(u"(\u2018|\u2019)", "'", q_text)
    q_text = re.sub('[^A-Za-z0-9]+', ' ', q_text).strip()

    MULTI_MATCH_QUERY = '{ "size":' + f'{TOP_K}' + ', "query": { "multi_match": { "query": ' + \
        f'"{q_text}"'+', "fields": [ "title", "headings", "content" ] } }}'
    return (MULTI_MATCH_QUERY, q_id)


def error_reporter(q_id):
    with open(f"{ERROR_FILE_NAME}", 'a') as ef:
        ef.write(str(q_id)+'\n')


# searching
def search(query):
    q_id = query[1]
    try:
        headers = {'Content-Type': 'application/json'}

        res = requests.get(
            f"http://elastic:{PASSWORD}@{HOSTNAME}:{PORT}/ms_marco/_search",
            headers=headers,
            data=query[0]
        ).json()

        if res['hits']['hits']:
            with open(f'{OUTPUT_FILE_NAME}', 'a') as f:
                for rank, hit in enumerate(res['hits']['hits']):
                    line = f'{q_id}\tQ0\t{hit["_id"]}\t{rank+1}\t{hit["_score"]}\t{RUN_NAME}\n'
                    f.write(line)

            return q_id
        else:
            error_reporter(q_id)
    except:
        error_reporter(q_id)


def bm25_search(q_id, q_text):
    QUERY, Q_ID = query_gen(q_id, q_text)

    result = search((QUERY, Q_ID))

    return result


train_df.progress_apply(lambda x: bm25_search(x['q_id'], x['q_text']), axis=1)

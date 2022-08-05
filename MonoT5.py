import gzip
import json
from tqdm import tqdm

# import modules
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
reranker = MonoT5()

# loading queries
QUERY_FILE = "./data/2022_queries.tsv"
with open(QUERY_FILE,'r') as qf:
    lines = qf.readlines()
    lines = [line.split() for line in lines]
    id2query = {line[0]:line[1] for line in lines}


# loading query's candidates
QUERY_RET_DOC_FILE = "./data/test_top_1k.tsv" 
q_docs = {}
with open(QUERY_RET_DOC_FILE,'r') as qtf:
    lines = qtf.readlines()
    for i in range(0,len(lines),1000):        
        docs = lines[i:i+1000]
        q_id = docs[0].split('\t')[0]
        docs = [line.split('\t')[2] for line in docs]
        q_docs[q_id] = docs

 
def get_document(document_id):
    """
    :return : dict_keys(['url', 'title', 'headings', 'body', 'docid'])
    """
    (string1, string2, bundlenum, position) = document_id.split('_')
    assert string1 == 'msmarco' and string2 == 'doc'
 
    with gzip.open(f'./data/msmarco_v2_doc/msmarco_doc_{bundlenum}.gz', 'rt', encoding='utf8') as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document['docid'] == document_id
        return document
 
def get_windows(text, window_size = 512,stride = 211):
    tokens = text.split()
    start = 0
    total_length = len(tokens)
    print(total_length)
    flag = True
    chunks = []
    while flag:
        end = start + window_size
        if end >= total_length:
            flag = False
            end = total_length

        chunks.append(tokens[start:end]) 
        
        start = start + stride
    return chunks
         

for q_id, doc_ids in q_docs.items():
    query = Query(id2query[q_id])
    for document_id in tqdm(doc_ids,leave=True):
        # keys = 'url', 'title', 'headings', 'body', 'docid'
        document_data = get_document(document_id) 
        doc = document_data['title'] +  ", " + document_data['headings'] + ", " + document_data['body']
        chunks = get_windows(doc,window_size=510,stride=510)
        chunks = [Text(text, {'docid': document_id + f"_{index}"}, 0)  for index,text in enumerate(chunks)]
        reranked = reranker.rerank(query, chunks)

        for i in range(len(reranked)):
            print(f'{i+1:2} {reranked[i].metadata["docid"]:15} {reranked[i].score:.5f} {reranked[i].metadata["result"]}')

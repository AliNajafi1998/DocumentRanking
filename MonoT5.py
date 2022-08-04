from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

# initializing the Model
reranker = MonoT5()


# need to read the queries

# need to read the retrieved quries

# slicing the documents to the passages

# reranking

# saving the results


# query = Query('who is Obama')


# passages = [["1", "obama is the president of USA"]]
# texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]
# reranked = reranker.rerank(query, texts)

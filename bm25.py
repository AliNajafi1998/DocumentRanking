from rank_bm25 import BM25Okapi, BM25, BM25L, BM25Plus


class CustomBM25:
    def __init__(self, corpus, bm25_type="okapi", tokenizer=str.split):
        self.corpus = corpus
        if bm25_type == 'bm25':
            self.bm25 = BM25(corpus=corpus, tokenizer=tokenizer)
        elif bm25_type == "okapi":
            self.bm25 = BM25Okapi(corpus=corpus, tokenizer=tokenizer)
        elif bm25_type == 'bm25l':
            self.bm25 = BM25L(corpus=corpus, tokenizer=tokenizer)
        elif bm25_type == 'bm25plus':
            self.bm25 = BM25Plus(corpus=corpus, tokenizer=tokenizer)
        else:
            raise Exception(f"There is no {bm25_type}")

    def get_top_k(self, query, k):
        return self.bm25.get_top_n(query.split(), self.corpus, n=k)

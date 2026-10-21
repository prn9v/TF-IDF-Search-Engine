"""
ir_engine.py - a from-scratch TF-IDF / Vector Space Model search engine.

Pipeline
--------
1. Pre-processing : lowercase -> tokenize -> stop-word removal -> Porter stemming
2. Indexing       : inverted index  term -> {doc_id: term frequency}
3. Weighting      : w(t,d) = (1 + ln tf) * idf(t),   idf(t) = ln((1+N)/(1+df)) + 1
4. Ranking        : cosine similarity between the query vector and each document vector
5. Evaluation     : Precision@k and Average Precision (MAP) against labelled categories

No external search library is used - everything above is implemented here.
"""
import math
import re
from collections import Counter, defaultdict
from html import escape

try:  # NLTK's Porter stemmer needs no extra data download
    from nltk.stem import PorterStemmer

    _STEMMER = PorterStemmer()
    STEMMER_AVAILABLE = True
except Exception:  # pragma: no cover - only when nltk is missing
    _STEMMER = None
    STEMMER_AVAILABLE = False

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can cannot could did do does doing don down during each
    few for from further had has have having he her here hers herself him himself his how i
    if in into is it its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you
    your yours yourself yourselves also may might shall us one get got like said say says
    use used using ve ll re
    """.split()
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Preprocessor:
    """Turns raw text into a list of normalised index terms."""

    def __init__(self, stem=True, remove_stopwords=True):
        self.stem = bool(stem) and STEMMER_AVAILABLE
        self.remove_stopwords = remove_stopwords
        self._cache = {}

    def _stem_word(self, word):
        stemmed = self._cache.get(word)
        if stemmed is None:
            stemmed = _STEMMER.stem(word)
            self._cache[word] = stemmed
        return stemmed

    def normalise(self, word):
        """Normalise ONE word. Returns None if the word should be dropped."""
        w = word.lower()
        if len(w) < 2 or len(w) > 25 or w.isdigit():
            return None
        if self.remove_stopwords and w in STOPWORDS:
            return None
        return self._stem_word(w) if self.stem else w

    def __call__(self, text):
        terms = []
        for token in TOKEN_RE.findall(text):
            term = self.normalise(token)
            if term:
                terms.append(term)
        return terms


class TfidfEngine:
    """Inverted index + TF-IDF weighting + cosine-similarity ranking."""

    def __init__(self, texts, stem=True, remove_stopwords=True):
        self.pre = Preprocessor(stem=stem, remove_stopwords=remove_stopwords)
        self.texts = list(texts)
        self.N = len(self.texts)

        # 1) inverted index: term -> {doc_id: raw term frequency}
        self.tf = defaultdict(dict)
        self.doc_len = []
        for doc_id, text in enumerate(self.texts):
            terms = self.pre(text)
            self.doc_len.append(len(terms))
            for term, count in Counter(terms).items():
                self.tf[term][doc_id] = count

        # 2) document frequency and inverse document frequency
        self.df = {term: len(postings) for term, postings in self.tf.items()}
        self.idf = {
            term: math.log((1 + self.N) / (1 + df)) + 1.0 for term, df in self.df.items()
        }

        # 3) TF-IDF weight of every (term, doc) pair + document vector lengths
        self.weights = {}
        norm_sq = [0.0] * self.N
        for term, postings in self.tf.items():
            idf = self.idf[term]
            term_weights = {}
            for doc_id, count in postings.items():
                w = (1.0 + math.log(count)) * idf
                term_weights[doc_id] = w
                norm_sq[doc_id] += w * w
            self.weights[term] = term_weights
        self.doc_norm = [math.sqrt(x) for x in norm_sq]

    # ------------------------------------------------------------------ query
    def query_vector(self, query):
        """Returns (processed tokens, {term: weight}, [terms not in the index])."""
        tokens = self.pre(query)
        vector, unknown = {}, []
        for term, count in Counter(tokens).items():
            if term in self.idf:
                vector[term] = (1.0 + math.log(count)) * self.idf[term]
            else:
                unknown.append(term)
        return tokens, vector, unknown

    def search(self, query, k=10, allowed=None):
        """
        Rank documents for `query` by cosine similarity.
        k=None returns every document that shares at least one term with the query.
        allowed = optional set of doc_ids the results must belong to.
        """
        tokens, qvec, unknown = self.query_vector(query)
        q_norm = math.sqrt(sum(w * w for w in qvec.values()))
        result = {"hits": [], "tokens": tokens, "qvec": qvec, "unknown": unknown}
        if q_norm == 0:
            return result

        dot = defaultdict(float)  # only documents that contain a query term are touched
        for term, wq in qvec.items():
            for doc_id, wd in self.weights[term].items():
                dot[doc_id] += wq * wd

        scored = [
            (value / (q_norm * self.doc_norm[doc_id]), doc_id)
            for doc_id, value in dot.items()
            if self.doc_norm[doc_id] > 0 and (allowed is None or doc_id in allowed)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        if k is not None:
            scored = scored[:k]
        result["hits"] = [(doc_id, score) for score, doc_id in scored]
        return result

    def explain(self, doc_id, qvec):
        """Per-term breakdown. The 'contribution' column sums to the cosine score."""
        q_norm = math.sqrt(sum(w * w for w in qvec.values()))
        rows = []
        for term, wq in qvec.items():
            wd = self.weights[term].get(doc_id)
            if wd is None:
                continue
            rows.append(
                {
                    "term": term,
                    "tf in doc": self.tf[term][doc_id],
                    "df": self.df[term],
                    "idf": round(self.idf[term], 3),
                    "query weight": round(wq, 3),
                    "doc weight": round(wd, 3),
                    "contribution": round(wq * wd / (q_norm * self.doc_norm[doc_id]), 4),
                }
            )
        rows.sort(key=lambda r: -r["contribution"])
        return rows

    # ------------------------------------------------------------- statistics
    def stats(self):
        return {
            "documents": self.N,
            "vocabulary": len(self.df),
            "avg_terms_per_doc": round(sum(self.doc_len) / max(self.N, 1), 1),
            "postings": sum(self.df.values()),
        }

    def top_terms_by_df(self, n=20):
        return sorted(self.df.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


# ------------------------------------------------------------------ snippets
def make_snippet(text, query_terms, pre, width=260):
    """HTML-safe snippet around the first match, with matching words in <mark>."""
    text = " ".join(text.split())
    matches = [
        (m.start(), m.end())
        for m in TOKEN_RE.finditer(text)
        if pre.normalise(m.group()) in query_terms
    ]
    first = matches[0][0] if matches else 0
    start = max(0, first - 60)
    end = min(len(text), start + width)

    parts, pos = [], start
    for s, e in matches:
        if e <= start:
            continue
        if s >= end:
            break
        s, e = max(s, start), min(e, end)
        parts.append(escape(text[pos:s]))
        parts.append("<mark>" + escape(text[s:e]) + "</mark>")
        pos = e
    parts.append(escape(text[pos:end]))
    return ("... " if start > 0 else "") + "".join(parts) + (" ..." if end < len(text) else "")


# ---------------------------------------------------------------- evaluation
def precision_at_k(ranked_ids, relevant, k):
    return sum(1 for d in ranked_ids[:k] if d in relevant) / k


def average_precision(ranked_ids, relevant):
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant) if relevant else 0.0


def evaluate(engine, labels, queries):
    """
    labels  : list with one category label per document (same order as the corpus)
    queries : list of (query text, relevant category)
    A document is relevant to a query when its category equals the query's category.
    """
    rows = []
    for query, category in queries:
        relevant = {i for i, label in enumerate(labels) if label == category}
        if not relevant:
            continue
        ranked = [doc_id for doc_id, _ in engine.search(query, k=None)["hits"]]
        rows.append(
            {
                "Query": query,
                "Relevant category": category,
                "Relevant docs": len(relevant),
                "P@5": round(precision_at_k(ranked, relevant, 5), 3),
                "P@10": round(precision_at_k(ranked, relevant, 10), 3),
                "AP": round(average_precision(ranked, relevant), 3),
            }
        )
    return rows
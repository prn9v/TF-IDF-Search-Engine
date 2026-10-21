# TF-IDF Search Engine (Information Retrieval mini project)

A search engine built from scratch with the **Vector Space Model**: inverted index, TF-IDF weighting,
cosine-similarity ranking, and evaluation (Precision@k, MAP). The UI is Streamlit.

## Project structure

```
ir-search/
├── app.py            # Streamlit UI (Search, Evaluation, Index inspector, How it works)
├── ir_engine.py      # Pre-processing, inverted index, TF-IDF, cosine ranking, metrics
├── corpus.py         # 20 Newsgroups loader + 36-document offline sample + test queries
├── requirements.txt
└── README.md
```

## Run it in VS Code (or any local machine)

1. Install Python 3.9+ and open an empty folder in VS Code.
2. Create the four code files above (copy them from the download) into that folder.
3. Open the terminal in VS Code and create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install the packages: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`
6. Your browser opens at http://localhost:8501. The first run with "20 Newsgroups" downloads the
   dataset (about 14 MB, needs internet). If the download fails, the app falls back to the offline sample.

## Run it in Google Colab

Upload `ir_search_colab.ipynb` to Colab and run the cells top to bottom (details are inside the notebook).

## How to build it step by step (what to explain in your report)

1. **Choose the collection.** 20 Newsgroups, 6 topics, up to 500 posts each. The topic label is
   used later as the relevance judgement.
2. **Pre-process** (`Preprocessor`): lowercase, tokenize with a regex, remove stop words, Porter-stem.
   Same function is applied to documents and queries, so they match.
3. **Build the inverted index** (`TfidfEngine.__init__`): `term -> {doc_id: term frequency}`.
4. **Compute document frequency and IDF**: `idf = ln((1+N)/(1+df)) + 1`.
5. **Weight terms**: `w = (1 + ln tf) * idf`; store each document's vector length for normalisation.
6. **Process a query**: same pre-processing, same weighting, giving a query vector.
7. **Rank**: walk only the posting lists of the query terms, accumulate dot products, divide by the
   vector lengths (cosine), sort descending.
8. **Explain results**: the per-term contribution table shows why a document scored as it did.
9. **Evaluate**: for each test query, relevant documents = documents of the matching category.
   Compute P@5, P@10, Average Precision, and MAP.
10. **Experiment**: compare MAP with no pre-processing, stop words only, and stop words + stemming.

## Things to try for your assignment screenshots

- Search `space shuttle orbit launch` and open "Why this rank?" on the top result.
- Index inspector: look up the term `orbit` to see its posting list.
- Evaluation tab: run the evaluation, then the pre-processing comparison.
- Use the sidebar to turn stemming off and watch the results change.

## Using your own documents

Choose "Upload my own CSV" in the sidebar. Required column: `text`. Optional: `title`, `category`
(the Evaluation tab needs `category`).
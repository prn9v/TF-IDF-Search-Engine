"""
app.py - Streamlit UI for the TF-IDF / Vector Space Model mini search engine.
Run with:   streamlit run app.py
"""
import time
from html import escape

import pandas as pd
import streamlit as st

from corpus import EVAL_QUERIES, SAMPLE_ROWS, load_newsgroups
from ir_engine import STEMMER_AVAILABLE, TfidfEngine, evaluate, make_snippet

st.set_page_config(page_title="TF-IDF Search Engine", page_icon="🔎", layout="wide")

SRC_NEWS = "20 Newsgroups (auto-download)"
SRC_SAMPLE = "Built-in sample (offline)"
SRC_CSV = "Upload my own CSV"


# ----------------------------------------------------------------- data loading
@st.cache_data(show_spinner="Downloading 20 Newsgroups (only the first time)...")
def cached_newsgroups(per_category):
    return load_newsgroups(per_category)


def get_dataframe(source, per_category, uploaded):
    """Returns (DataFrame with title/text/category, human readable label)."""
    if source == SRC_NEWS:
        try:
            return pd.DataFrame(cached_newsgroups(per_category)), f"20 Newsgroups ({per_category}/topic)"
        except Exception as exc:
            st.warning(
                f"Could not download 20 Newsgroups ({type(exc).__name__}). "
                "Using the built-in sample corpus instead."
            )
            return pd.DataFrame(SAMPLE_ROWS), "Built-in sample (fallback)"

    if source == SRC_SAMPLE:
        return pd.DataFrame(SAMPLE_ROWS), "Built-in sample"

    # CSV upload
    if uploaded is None:
        st.info("Upload a CSV with a **text** column (optional: **title**, **category**) in the sidebar.")
        st.stop()
    df = pd.read_csv(uploaded, encoding_errors="replace")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "text" not in df.columns:
        st.error("Your CSV needs a column named 'text'.")
        st.stop()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        st.error("The CSV has no non-empty text rows.")
        st.stop()
    if "title" not in df.columns:
        df["title"] = df["text"].str.slice(0, 70)
    df["title"] = df["title"].fillna("").astype(str)
    if "category" in df.columns:
        df["category"] = df["category"].fillna("unknown").astype(str)
    return df, f"Uploaded CSV ({uploaded.name})"


@st.cache_resource(show_spinner="Building the inverted index...")
def get_engine(dataset_key, _texts, stem, remove_stop):
    # dataset_key makes the cache distinguish datasets; _texts is not hashed.
    return TfidfEngine(_texts, stem=stem, remove_stopwords=remove_stop)


# ---------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
source = st.sidebar.radio("Document collection", [SRC_NEWS, SRC_SAMPLE, SRC_CSV])
per_category = 200
uploaded = None
if source == SRC_NEWS:
    per_category = st.sidebar.slider("Documents per topic", 50, 500, 200, step=50)
elif source == SRC_CSV:
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

st.sidebar.subheader("Pre-processing")
use_stem = st.sidebar.checkbox("Porter stemming", value=True, disabled=not STEMMER_AVAILABLE)
use_stop = st.sidebar.checkbox("Remove stop words", value=True)
if not STEMMER_AVAILABLE:
    st.sidebar.warning("nltk not installed - stemming is off. Run: pip install nltk")
top_k = st.sidebar.slider("Results to show", 3, 30, 10)

# ------------------------------------------------------------------ build index
df, label = get_dataframe(source, per_category, uploaded)
has_cat = "category" in df.columns
data_hash = int(pd.util.hash_pandas_object(df["text"], index=False).sum())
dataset_key = f"{label}|{len(df)}|{data_hash}"

engine = get_engine(dataset_key, df["text"].tolist(), use_stem, use_stop)

st.title("TF-IDF Search Engine")
st.caption(f"Vector Space Model with cosine similarity - collection: **{label}**, {len(df):,} documents")

tab_search, tab_eval, tab_index, tab_how = st.tabs(
    ["Search", "Evaluation", "Index inspector", "How it works"]
)

# ------------------------------------------------------------------ tab: search
with tab_search:
    st.session_state.setdefault("query", "")

    def _set_query(text):
        st.session_state["query"] = text

    if source != SRC_CSV:
        examples = ["space shuttle orbit launch", "hockey playoff goalie", "encryption private key", "car engine brakes"]
        st.write("Try an example:")
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            col.button(ex, on_click=_set_query, args=(ex,), key=f"ex_{ex}")

    query = st.text_input("Search query", key="query", placeholder="e.g. doctor treatment for high blood pressure")

    allowed = None
    if has_cat:
        chosen = st.multiselect("Only search these categories (optional)", sorted(df["category"].unique()))
        if chosen:
            allowed = set(df.index[df["category"].isin(chosen)])

    if query.strip():
        t0 = time.perf_counter()
        res = engine.search(query, k=top_k, allowed=allowed)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        st.caption(
            "Processed query terms: "
            + (", ".join(f"`{t}`" for t in sorted(set(res["tokens"]))) or "none (only stop words?)")
        )
        if res["unknown"]:
            st.warning("Not in the index (ignored): " + ", ".join(res["unknown"]))

        if not res["hits"]:
            st.info("No document shares a term with this query.")
        else:
            st.write(f"**{len(res['hits'])} results** shown in {elapsed_ms:.1f} ms")
            query_terms = set(res["qvec"])
            for rank, (doc_id, score) in enumerate(res["hits"], start=1):
                row = df.iloc[doc_id]
                cat_html = f" &nbsp;|&nbsp; {escape(str(row['category']))}" if has_cat else ""
                snippet = make_snippet(row["text"], query_terms, engine.pre)
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size:1.05rem;font-weight:600'>{rank}. {escape(str(row['title']))}</div>"
                        f"<div style='opacity:.7;font-size:.85rem'>cosine score <b>{score:.4f}</b>{cat_html}</div>"
                        f"<div style='margin-top:.4rem'>{snippet}</div>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Why this rank? (per-term contribution)"):
                        st.dataframe(pd.DataFrame(engine.explain(doc_id, res["qvec"])), hide_index=True)
                        st.caption("The contribution column adds up to the cosine score.")

# ------------------------------------------------------------- tab: evaluation
with tab_eval:
    st.subheader("How good is the ranking?")
    if not has_cat:
        st.info("Evaluation needs a **category** column: a document counts as relevant when its category matches the query's category.")
    else:
        st.write(
            "A document is **relevant** to a query when its category equals the category you give the query. "
            "Edit the list below (one per line: `query => category`)."
        )
        known = set(df["category"])
        default_lines = (
            "\n".join(f"{q} => {c}" for q, c in EVAL_QUERIES if c in known) if source != SRC_CSV else ""
        )
        text = st.text_area("Test queries", value=default_lines, height=230, key=f"evalq|{dataset_key}")
        queries = []
        for line in text.splitlines():
            if "=>" in line:
                q, c = line.split("=>", 1)
                if q.strip() and c.strip():
                    queries.append((q.strip(), c.strip()))

        run_key = f"{dataset_key}|{use_stem}|{use_stop}|{text}"
        if st.button("Run evaluation"):
            st.session_state["eval"] = {"key": run_key, "rows": evaluate(engine, df["category"].tolist(), queries)}
        ev = st.session_state.get("eval")
        if ev and ev["key"] == run_key:
            if not ev["rows"]:
                st.warning("No valid test queries (check the category names).")
            else:
                res_df = pd.DataFrame(ev["rows"])
                c1, c2, c3 = st.columns(3)
                c1.metric("MAP (mean average precision)", f"{res_df['AP'].mean():.3f}")
                c2.metric("Mean P@5", f"{res_df['P@5'].mean():.3f}")
                c3.metric("Mean P@10", f"{res_df['P@10'].mean():.3f}")
                st.dataframe(res_df, hide_index=True)
                st.bar_chart(res_df.set_index("Query")[["P@10", "AP"]])

        st.divider()
        st.write("**Experiment:** does pre-processing help? (builds 3 indexes, takes a few seconds)")
        cmp_key = f"{dataset_key}|{text}"
        if st.button("Compare pre-processing settings"):
            rows = []
            configs = [
                ("No stop-word removal, no stemming", False, False),
                ("Stop-word removal only", False, True),
                ("Stop-word removal + stemming", True, True),
            ]
            with st.spinner("Building and evaluating 3 indexes..."):
                for name, stem_flag, stop_flag in configs:
                    t0 = time.perf_counter()
                    eng = TfidfEngine(df["text"].tolist(), stem=stem_flag, remove_stopwords=stop_flag)
                    build_s = time.perf_counter() - t0
                    ev_rows = evaluate(eng, df["category"].tolist(), queries)
                    if not ev_rows:
                        continue
                    rows.append(
                        {
                            "Setting": name,
                            "Vocabulary size": len(eng.df),
                            "Build time (s)": round(build_s, 2),
                            "MAP": round(sum(r["AP"] for r in ev_rows) / len(ev_rows), 3),
                            "Mean P@10": round(sum(r["P@10"] for r in ev_rows) / len(ev_rows), 3),
                        }
                    )
            st.session_state["cmp"] = {"key": cmp_key, "rows": rows}
        cmp = st.session_state.get("cmp")
        if cmp and cmp["key"] == cmp_key and cmp["rows"]:
            cmp_df = pd.DataFrame(cmp["rows"])
            st.dataframe(cmp_df, hide_index=True)
            st.bar_chart(cmp_df.set_index("Setting")[["MAP", "Mean P@10"]])

# ------------------------------------------------------------ tab: index viewer
with tab_index:
    st.subheader("Inside the inverted index")
    stats = engine.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents (N)", f"{stats['documents']:,}")
    c2.metric("Vocabulary size", f"{stats['vocabulary']:,}")
    c3.metric("Avg terms / doc", stats["avg_terms_per_doc"])
    c4.metric("Postings", f"{stats['postings']:,}")

    st.write("**Most frequent terms (by document frequency)**")
    st.dataframe(
        pd.DataFrame(
            [{"term": t, "df": d, "idf": round(engine.idf[t], 3)} for t, d in engine.top_terms_by_df(20)]
        ),
        hide_index=True,
    )

    lookup = st.text_input("Look up a term's posting list", placeholder="e.g. orbit")
    if lookup.strip():
        terms = engine.pre(lookup)
        if not terms:
            st.info("That word is removed by pre-processing (stop word?).")
        for term in dict.fromkeys(terms):
            if term not in engine.tf:
                st.warning(f"`{term}` is not in the index.")
                continue
            st.write(f"**`{term}`** - appears in {engine.df[term]} documents, idf = {engine.idf[term]:.3f}")
            postings = sorted(engine.weights[term].items(), key=lambda kv: -kv[1])[:15]
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "doc_id": d,
                            "title": df.iloc[d]["title"],
                            "tf": engine.tf[term][d],
                            "tf-idf weight": round(w, 3),
                        }
                        for d, w in postings
                    ]
                ),
                hide_index=True,
            )

# ---------------------------------------------------------------- tab: how it is
with tab_how:
    st.subheader("How the ranking works")
    st.markdown(
        """
1. **Pre-process** every document: lowercase, split into words, drop stop words, stem (`launching` -> `launch`).
2. **Build an inverted index**: for each term, the list of documents containing it and the term frequency.
3. **Weight** each term in each document with TF-IDF:
"""
    )
    st.latex(r"w_{t,d} = (1 + \ln tf_{t,d}) \times idf_t \qquad idf_t = \ln\frac{1+N}{1+df_t} + 1")
    st.markdown("4. Turn the **query** into a vector the same way, then rank documents by **cosine similarity**:")
    st.latex(r"\cos(q,d) = \frac{\sum_t w_{t,q}\, w_{t,d}}{\lVert q \rVert \; \lVert d \rVert}")
    st.markdown(
        """
Only documents that contain at least one query term are scored, because the index
lets us walk the posting lists of the query terms instead of scanning every document.

**Evaluation** uses category labels as relevance judgements:
*Precision@k* = fraction of the top k results that are relevant;
*Average Precision* rewards putting relevant documents early in the ranking;
*MAP* is the mean of AP over all test queries.
"""
    )
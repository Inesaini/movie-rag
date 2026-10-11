"""Streamlit UI for the movie RAG.   Run:  streamlit run app.py"""
import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_core import INDEX_DIR, Engine, LLMConfig, answer, log_event, read_log, stem

st.set_page_config(page_title="Tip of the Tongue", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading index…")
def load_engine(index_dir: str) -> Engine:
    return Engine(index_dir)


if not (Path(INDEX_DIR) / "meta.json").exists():
    st.error("No index found. Build it first:\n\n`python build_index.py --csv data/wiki_movie_plots_deduped.csv`")
    st.stop()
engine = load_engine(str(INDEX_DIR))
movies = engine.movies


# ----------------------------------------------------------------------------- helpers
def highlight(text: str, query: str, max_chars: int = 700) -> str:
    """Escape the passage and wrap words that match the query (after stemming) in <mark>."""
    q_stems = {stem(w) for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    text = text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"
    def repl(m):
        w = m.group(0)
        return f"<mark>{w}</mark>" if stem(w.lower()) in q_stems else w
    return re.sub(r"[A-Za-z0-9]+", repl, html.escape(text))


def film_line(mid: int) -> str:
    m = movies.loc[mid]
    return f"**{m.title}** ({m.year}) · {m.origin} · dir. {m.director} · _{m.genre}_"


def show_info(info, applied, timings):
    cols = st.columns(4)
    cols[0].metric("Dense weight α", f"{info.alpha:.2f}")
    cols[1].metric("Filters", ", ".join(f"{k}={v}" for k, v in applied.items() if k != "movie_ids") or "none")
    cols[2].metric("Retrieval", f"{timings.get('total_ms', timings.get('bm25_ms', 0) + timings.get('dense_ms', 0)):.0f} ms")
    if "llm_ms" in timings:
        cols[3].metric("LLM", f"{timings['llm_ms'] / 1000:.1f} s")
    st.caption(f"Query analyzer: {info.reason}" + (f" · people detected: {', '.join(info.people)}" if info.people else ""))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🎬 Tip of the Tongue")
    st.caption(f"{len(movies):,} films · {len(engine.chunks):,} passages · {engine.model_name.split('/')[-1]}")

    st.subheader("Retrieval")
    mode = st.selectbox("Mode", ["adaptive", "hybrid", "dense", "bm25"], index=0,
                        help="adaptive: the query analyzer picks the weight and filters")
    alpha = st.slider("Dense weight α (hybrid mode)", 0.0, 1.0, 0.5, 0.05, disabled=mode != "hybrid")
    top_k = st.slider("Results", 3, 20, 8)
    use_graph = st.toggle("Graph boost (people named in the query)", value=True)

    st.subheader("Filters")
    y0, y1 = int(movies.year.min()), int(movies.year.max())
    years = st.slider("Release year", y0, y1, (y0, y1))
    origins = st.multiselect("Origin", sorted(movies.origin.unique()))
    genre_opts = [""] + sorted(engine.analyze.genres)
    genre = st.selectbox("Genre", genre_opts, format_func=lambda g: g or "any")

    st.subheader("Answer generation")
    backend = st.selectbox("LLM backend", ["ollama", "openai", "none"],
                           help="'openai' works with any OpenAI-compatible server (LM Studio, vLLM, Groq, …)")
    default_url = {"ollama": "http://localhost:11434", "openai": "https://api.openai.com/v1", "none": ""}[backend]
    llm_cfg = LLMConfig(backend=backend,
                        model=st.text_input("Model", "llama3.1:8b" if backend == "ollama" else "gpt-4o-mini",
                                            disabled=backend == "none"),
                        url=st.text_input("Server URL", default_url, disabled=backend == "none"),
                        api_key=st.text_input("API key", type="password", disabled=backend != "openai"))

ui_filters = {"years": years if years != (y0, y1) else None, "origin": origins or None, "genre": genre or None}
search_kw = dict(mode=mode, alpha=alpha, ui_filters=ui_filters)

tab_find, tab_ask, tab_monitor = st.tabs(["🔎 Find a movie", "💬 Ask about a movie", "📊 Monitoring"])

# ----------------------------------------------------------------------------- tab 1: find
with tab_find:
    st.markdown("Describe a film you half-remember. Names, decades, origins and genres in your "
                "sentence are picked up automatically.")
    examples = ["a man keeps reliving the same day", "a 90s horror movie about a cursed videotape",
                "Un film où un robot tombe amoureux", "a film by Christopher Nolan about dreams"]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["find_q"] = ex
    query = st.text_input("Your description", key="find_q", placeholder="the movie where…")

    if query:
        results, info, applied, timings = engine.search_movies(query, k=top_k, use_graph=use_graph, **search_kw)
        log_event({"tab": "find", "query": query, "mode": mode, "alpha": info.alpha, "filters": applied,
                   "n_results": len(results), "top_movie": int(results[0].movie_id) if results else None, **timings})
        show_info(info, applied, timings)
        if not results:
            st.warning("Nothing matched. Try loosening the filters.")
        for rank, r in enumerate(results, 1):
            with st.container(border=True):
                left, right = st.columns([5, 1])
                left.markdown(f"{rank}. {film_line(r.movie_id)}")
                right.markdown(f"`{r.score:.3f}`" + (" 🕸️" if r.graph_bonus else ""),
                               help="🕸️ = boosted by the knowledge graph")
                st.markdown(f"<div style='font-size:0.92em'>{highlight(r.best_chunk['body'], query)}</div>",
                            unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                with b1.popover("Full plot"):
                    st.write(movies.loc[r.movie_id].plot)
                    if isinstance(movies.loc[r.movie_id].url, str):
                        st.link_button("Wikipedia", movies.loc[r.movie_id].url)
                with b2.popover("More like this"):
                    st.dataframe(engine.more_like_this(r.movie_id).drop(columns="movie_id"),
                                 hide_index=True, use_container_width=True)
                if b3.button("Ask about it →", key=f"ask_{r.movie_id}"):
                    st.session_state["ask_movie"] = int(r.movie_id)
                    st.toast(f"Selected {movies.loc[r.movie_id].title} - open the 'Ask about a movie' tab")

# ----------------------------------------------------------------------------- tab 2: ask
with tab_ask:
    labels = {None: "All films", **{int(m.movie_id): f"{m.title} ({m.year})"
                                    for m in movies.sort_values(["title", "year"]).itertuples()}}
    options = list(labels)
    current = st.session_state.get("ask_movie")
    chosen = st.selectbox("Film (optional - leave empty to search all plots)", options,
                          index=options.index(current) if current in labels else 0, format_func=labels.get)
    st.session_state["ask_movie"] = chosen
    question = st.text_input("Question", placeholder="Who betrays the hero at the end?", key="ask_q")

    if question:
        with st.spinner("Retrieving and generating…"):
            out = answer(engine, question, llm_cfg, movie_ids=[chosen] if chosen is not None else None, k=5)
        log_event({"tab": "ask", "query": question, "movie": chosen, "backend": backend, "alpha": out["info"].alpha,
                   "n_hits": len(out["hits"]), "error": out["error"],
                   "bad_citations": len(out["bad_citations"]), **out["timings"]})
        show_info(out["info"], {}, out["timings"])

        if out["answer"]:
            body = re.sub(r"\[(\d+#\d+)\]", r"`[\1]`", out["answer"])
            st.success(body)
            if out["bad_citations"]:
                st.warning(f"⚠️ The answer cites passages that weren't retrieved: {', '.join(out['bad_citations'])}")
            else:
                st.caption("✅ every citation points to a retrieved passage")
        elif out["error"]:
            st.error(out["error"] + " Showing the retrieved passages only.")
        elif backend == "none":
            st.info("No LLM selected - showing the most relevant passages.")

        st.markdown("#### Sources")
        for c, s in out["hits"]:
            with st.expander(f"`[{c['chunk_id']}]` {movies.loc[c['movie_id']].title} ({movies.loc[c['movie_id']].year}) · score {s:.3f}"):
                st.markdown(highlight(c["body"], question, max_chars=3000), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- tab 3: monitoring
with tab_monitor:
    logs = read_log()
    if logs.empty:
        st.info("No queries logged yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", len(logs))
        c2.metric("Median retrieval", f"{logs['total_ms'].median():.0f} ms" if "total_ms" in logs else "–")
        c3.metric("p95 retrieval", f"{logs['total_ms'].quantile(.95):.0f} ms" if "total_ms" in logs else "–")
        if "llm_ms" in logs and logs["llm_ms"].notna().any():
            c4.metric("Median LLM", f"{logs['llm_ms'].median() / 1000:.1f} s")
        stage_cols = [c for c in ("bm25_ms", "dense_ms", "total_ms", "llm_ms") if c in logs]
        st.markdown("**Latency per stage (ms)**")
        st.line_chart(logs[stage_cols].reset_index(drop=True))
        if "mode" in logs:
            st.markdown("**Queries by mode**")
            st.bar_chart(logs["mode"].value_counts())
        st.markdown("**Recent queries**")
        show = [c for c in ("ts", "tab", "query", "mode", "alpha", "filters", "total_ms", "llm_ms", "error") if c in logs]
        st.dataframe(logs[show].tail(50).iloc[::-1], hide_index=True, use_container_width=True)
        report = Path(INDEX_DIR) / "validation_report.csv"
        if report.exists():
            st.markdown("**Ingestion: document validation report**")
            rep = pd.read_csv(report)
            st.bar_chart(rep["status"].value_counts())

"""Core RAG engine: validation, chunking, BM25, dense retrieval, adaptive hybrid search,
metadata filters, knowledge graph and grounded generation.

Used by build_index.py (offline) and app.py (Streamlit UI).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests

log = logging.getLogger("movie_rag")

INDEX_DIR = Path("index")
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
PREFIXES = {  # (query prefix, passage prefix) expected by each model family
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "BAAI/bge-m3": ("", ""),
}

# --------------------------------------------------------------------------- validation
MOJIBAKE = re.compile(r"(â€|Ã©|Ã¨|Ã¢|Â )")
CITE = re.compile(r"\[\d+\]|\[citation needed\]|\[edit\]", re.I)
COLUMN_MAP = {"Release Year": "year", "Title": "title", "Origin/Ethnicity": "origin", "Director": "director",
              "Cast": "cast", "Genre": "genre", "Wiki Page": "url", "Plot": "plot"}


def clean_plot(s) -> tuple[str, bool, int]:
    s = "" if pd.isna(s) else str(s)
    fixed = False
    if MOJIBAKE.search(s):
        try:
            s, fixed = s.encode("cp1252").decode("utf-8"), True
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    n_cites = len(CITE.findall(s))
    s = CITE.sub("", s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s).strip()
    return s, fixed, n_cites


def validate(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean every plot, profile it, score its quality and mark what to drop."""
    df = raw.rename(columns=COLUMN_MAP).copy()
    cleaned = df["plot"].map(clean_plot)
    df["plot"] = cleaned.map(lambda t: t[0])
    df["mojibake_fixed"] = cleaned.map(lambda t: t[1])
    df["n_citations"] = cleaned.map(lambda t: t[2])
    df["n_words"] = df["plot"].str.split().str.len().fillna(0).astype(int)
    for col in ("director", "cast", "genre", "origin"):
        df[col] = df[col].fillna("unknown").astype(str).str.strip()
    df["missing_meta"] = df[["director", "cast", "genre"]].apply(
        lambda r: sum(v.lower() == "unknown" for v in r), axis=1)
    alpha_ratio = df["plot"].map(lambda s: sum(c.isalpha() for c in s) / max(len(s), 1))
    df["quality"] = (0.4 * np.clip(df["n_words"] / 150, 0, 1)
                     + 0.3 * np.clip((alpha_ratio - 0.6) / 0.2, 0, 1)
                     + 0.3 * (1 - df["missing_meta"] / 3)).round(3)
    df["status"] = np.select(
        [df["n_words"] < 25, df.duplicated("plot", keep="first"), df["quality"] < 0.6],
        ["drop: too short", "drop: duplicate plot", "flag: low quality"], default="ok")
    return df


# --------------------------------------------------------------------------- chunking
def header(m) -> str:
    return f"{m.title} ({m.year}) · {m.origin} · dir. {m.director} · {m.genre}"


def chunk_paragraph(plot: str, min_w: int = 40, max_w: int = 220) -> list[str]:
    out, buf = [], []
    for p in plot.split("\n\n"):
        buf += p.split()
        if len(buf) >= min_w:
            out += [" ".join(buf[i:i + max_w]) for i in range(0, len(buf), max_w)]
            buf = []
    if buf:
        if out and len(buf) < min_w:
            out[-1] += " " + " ".join(buf)
        else:
            out.append(" ".join(buf))
    return out


def build_chunks(movies: pd.DataFrame) -> list[dict]:
    chunks = []
    for m in movies.itertuples():
        for j, body in enumerate(chunk_paragraph(m.plot)):
            chunks.append({"chunk_id": f"{m.movie_id}#{j}", "movie_id": int(m.movie_id),
                           "text": f"{header(m)}\n{body}", "body": body})
    return chunks


# --------------------------------------------------------------------------- BM25
STOP = set(("a an the of to in on for is are be was were it its his her their they them he she him "
            "this that these those with by as at or and from than then there into who whom which what "
            "when where how after before while but not no so up out about over also has have had").split())


def stem(t: str) -> str:
    if len(t) > 4:
        for suf in ("ing", "ies", "es", "ed", "s"):
            if t.endswith(suf):
                return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        tfs = [Counter(tokenize(d)) for d in docs]
        self.N = len(docs)
        self.doclen = np.array([sum(c.values()) for c in tfs], dtype=np.float32)
        self.avg = float(self.doclen.mean()) if self.N else 1.0
        post = defaultdict(lambda: ([], []))
        for i, c in enumerate(tfs):
            for t, f in c.items():
                post[t][0].append(i)
                post[t][1].append(f)
        self.post = {t: (np.array(d), np.array(f, dtype=np.float32)) for t, (d, f) in post.items()}
        self.idf = {t: math.log(1 + (self.N - len(d) + 0.5) / (len(d) + 0.5)) for t, (d, _) in self.post.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(tokenize(query)):
            if t in self.post:
                d, f = self.post[t]
                s[d] += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.doclen[d] / self.avg))
        return s


# --------------------------------------------------------------------------- dense
_models: dict = {}


def get_model(name: str):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name, trust_remote_code=True)
    return _models[name]


def embed_passages(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    _, dp = PREFIXES.get(model_name, ("", ""))
    return get_model(model_name).encode([dp + t for t in texts], batch_size=batch_size,
                                        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)


def embed_query(q: str, model_name: str) -> np.ndarray:
    qp, _ = PREFIXES.get(model_name, ("", ""))
    return get_model(model_name).encode([qp + q], normalize_embeddings=True)[0].astype(np.float32)


# --------------------------------------------------------------------------- query analysis
FR_HINTS = set("le la les un une des du est qui que où dans avec pour sur film homme femme son sa ses il elle".split())


@dataclass
class QueryInfo:
    alpha: float
    people: list = field(default_factory=list)
    quoted: list = field(default_factory=list)
    non_english: bool = False
    filters: dict = field(default_factory=dict)
    reason: str = ""


class QueryAnalyzer:
    """Picks the dense/BM25 weight and extracts metadata filters. Vocabularies come from the data."""

    def __init__(self, movies: pd.DataFrame):
        self.people = set()
        for col in ("director", "cast"):
            for v in movies[col]:
                for name in re.split(r",| and ", v):
                    name = name.strip().lower()
                    if name != "unknown" and 2 <= len(name.split()) <= 4:
                        self.people.add(name)
        self.origins = {o.lower().replace("_", " "): o for o in movies.origin.unique()}
        self.genres = [g for g, _ in Counter(movies.genre.str.lower()).most_common(40)
                       if g != "unknown" and len(g.split()) == 1]

    def __call__(self, q: str) -> QueryInfo:
        low = q.lower()
        words = re.findall(r"[a-z0-9àâçéèêëîïôûùüœ']+", low)
        grams = {" ".join(words[i:i + n]) for n in (2, 3) for i in range(len(words) - n + 1)}
        people = sorted(g for g in grams if g in self.people)
        quoted = re.findall(r'"([^"]+)"', q)
        non_en = sum(w in FR_HINTS for w in words) >= 3 or bool(re.search(r"[àâçéèêîôûœ]", low))
        filt = {}
        if m := re.search(r"\b(19|20)(\d)0s\b", low):
            d = int(m.group(1) + m.group(2) + "0"); filt["years"] = (d, d + 9)
        elif m := re.search(r"\b(?:the\s+)?([2-9]0)s\b", low):
            d = 1900 + int(m.group(1)); filt["years"] = (d, d + 9)
        elif m := re.search(r"\b(?:in|from)\s+((?:19|20)\d\d)\b", low):
            filt["years"] = (int(m.group(1)), int(m.group(1)))
        if o := next((self.origins[k] for k in self.origins if re.search(rf"\b{re.escape(k)}\b", low)), None):
            filt["origin"] = o
        if g := next((g for g in self.genres if re.search(rf"\b{re.escape(g)}\b", low)), None):
            filt["genre"] = g
        if people or quoted:
            alpha, reason = 0.3, "names a person or quotes a title → favour exact matching (BM25)"
        elif non_en:
            alpha, reason = 0.9, "not English → BM25 can't match across languages, favour dense"
        else:
            alpha, reason = 0.6, "vague description → favour semantic (dense) matching"
        return QueryInfo(alpha, people, quoted, non_en, filt, reason)


# --------------------------------------------------------------------------- engine
def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


@dataclass
class SearchResult:
    movie_id: int
    score: float
    best_chunk: dict
    graph_bonus: float = 0.0


class Engine:
    def __init__(self, index_dir: Path | str = INDEX_DIR):
        index_dir = Path(index_dir)
        t0 = time.time()
        meta = json.loads((index_dir / "meta.json").read_text())
        self.model_name = meta["model"]
        self.movies = pd.read_pickle(index_dir / "movies.pkl").set_index("movie_id", drop=False)
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.emb = np.load(index_dir / "embeddings.npy")
        assert len(self.emb) == len(self.chunks), "embeddings and chunks out of sync - rebuild the index"
        self.movie_of = np.array([c["movie_id"] for c in self.chunks])
        self.bm25 = BM25([c["text"] for c in self.chunks])
        m = self.movies.loc[self.movie_of]
        self.c_year, self.c_origin = m.year.values, m.origin.values
        self.c_genre = np.array(m.genre.str.lower().tolist(), dtype=str)
        self.analyze = QueryAnalyzer(self.movies)
        self.graph = self._build_graph()
        log.info("engine loaded: %d films, %d chunks in %.1fs", len(self.movies), len(self.chunks), time.time() - t0)

    # ---- graph
    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for m in self.movies.itertuples():
            G.add_node(("movie", m.movie_id))
            for col in ("director", "cast", "genre"):
                kind = "genre" if col == "genre" else "person"
                for v in re.split(r",| and |/", getattr(m, col)):
                    v = v.strip().lower()
                    if v and v != "unknown":
                        G.add_edge(("movie", m.movie_id), (kind, v), rel=col)
        return G

    def more_like_this(self, movie_id: int, k: int = 5) -> pd.DataFrame:
        pr = nx.pagerank(self.graph, personalization={("movie", movie_id): 1.0}, alpha=0.85, max_iter=500, tol=1e-8)
        top = sorted(((v, n[1]) for n, v in pr.items() if n[0] == "movie" and n[1] != movie_id), reverse=True)[:k]
        return self.movies.loc[[mid for _, mid in top], ["movie_id", "title", "year", "director", "genre"]]

    # ---- filters
    def filter_mask(self, filt: dict) -> np.ndarray:
        mask = np.ones(len(self.chunks), bool)
        if filt.get("years"):
            mask &= (self.c_year >= filt["years"][0]) & (self.c_year <= filt["years"][1])
        if filt.get("origin"):
            origins = filt["origin"] if isinstance(filt["origin"], (list, tuple)) else [filt["origin"]]
            mask &= np.isin(self.c_origin, origins)
        if filt.get("genre"):
            mask &= np.char.find(self.c_genre, filt["genre"].lower()) >= 0
        if filt.get("movie_ids") is not None:
            mask &= np.isin(self.movie_of, filt["movie_ids"])
        return mask

    # ---- scoring
    def chunk_scores(self, query: str, mode: str = "adaptive", alpha: float | None = None,
                     ui_filters: dict | None = None, auto_filters: bool = True, timings: dict | None = None):
        timings = timings if timings is not None else {}
        info = self.analyze(query)
        t = time.perf_counter()
        bm = minmax(self.bm25.scores(query)) if mode != "dense" else None
        timings["bm25_ms"] = 1000 * (time.perf_counter() - t)
        t = time.perf_counter()
        de = minmax(self.emb @ embed_query(query, self.model_name)) if mode != "bm25" else None
        timings["dense_ms"] = 1000 * (time.perf_counter() - t)
        if mode == "bm25":
            s, a = bm, 0.0
        elif mode == "dense":
            s, a = de, 1.0
        else:
            a = alpha if (mode == "hybrid" and alpha is not None) else info.alpha
            s = a * de + (1 - a) * bm
        info.alpha = a
        filt = dict(info.filters) if (auto_filters and mode == "adaptive") else {}
        filt.update({k: v for k, v in (ui_filters or {}).items() if v})   # UI filters win
        applied = {}
        if filt:
            mask = self.filter_mask(filt)
            if mask.any():
                s = np.where(mask, s, -1.0)
                applied = filt
            else:
                info.reason += " · filters matched nothing, ignored"
        return s, info, applied

    def search_movies(self, query: str, k: int = 10, use_graph: bool = True, graph_bonus: float = 0.5, **kw):
        timings: dict = {}
        t0 = time.perf_counter()
        s, info, applied = self.chunk_scores(query, timings=timings, **kw)
        order = np.argsort(-s)
        best_chunk, best_score = {}, {}
        for i in order:                          # max-pool chunk scores per film
            if s[i] < 0:
                break
            mid = int(self.movie_of[i])
            if mid not in best_score:
                best_score[mid], best_chunk[mid] = float(s[i]), self.chunks[i]
            if len(best_score) >= k * 5:
                break
        bonus = defaultdict(float)
        if use_graph:
            for person in info.people:
                node = ("person", person)
                if node in self.graph:
                    for _, mid in self.graph.neighbors(node):
                        bonus[mid] += graph_bonus
            for mid in bonus:                    # graph can surface films the text search missed
                if mid not in best_score:
                    idx = np.where(self.movie_of == mid)[0]
                    if len(idx) and s[idx].max() >= 0:
                        j = idx[np.argmax(s[idx])]
                        best_score[mid], best_chunk[mid] = float(s[j]), self.chunks[j]
        ranked = sorted(best_score, key=lambda m: -(best_score[m] + bonus.get(m, 0)))[:k]
        timings["total_ms"] = 1000 * (time.perf_counter() - t0)
        results = [SearchResult(m, best_score[m] + bonus.get(m, 0), best_chunk[m], bonus.get(m, 0)) for m in ranked]
        return results, info, applied, timings

    def search_chunks(self, query: str, k: int = 5, movie_ids=None, **kw):
        timings: dict = {}
        ui = dict(kw.pop("ui_filters", {}) or {})
        if movie_ids is not None:
            ui["movie_ids"] = list(movie_ids)
        s, info, applied = self.chunk_scores(query, ui_filters=ui, auto_filters=movie_ids is None,
                                             timings=timings, **kw)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top if s[i] >= 0], info, timings


# --------------------------------------------------------------------------- generation
PROMPT = """Answer the question using ONLY the movie plot passages below.
Cite the passages you use as [id]. If the answer isn't in the passages, say "I don't know".
Answer briefly, in the language of the question.

PASSAGES:
{context}

QUESTION: {question}
ANSWER:"""


@dataclass
class LLMConfig:
    backend: str = "ollama"                       # 'ollama' | 'openai' (any OpenAI-compatible server) | 'none'
    model: str = "llama3.1:8b"
    url: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.0


def call_llm(prompt: str, cfg: LLMConfig) -> str:
    if cfg.backend == "ollama":
        r = requests.post(f"{cfg.url.rstrip('/')}/api/generate", timeout=180,
                          json={"model": cfg.model, "prompt": prompt, "stream": False,
                                "options": {"temperature": cfg.temperature}})
        r.raise_for_status()
        return r.json()["response"].strip()
    if cfg.backend == "openai":
        r = requests.post(f"{cfg.url.rstrip('/')}/chat/completions", timeout=180,
                          headers={"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {},
                          json={"model": cfg.model, "temperature": cfg.temperature,
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise ValueError("no LLM backend")


def answer(engine: Engine, question: str, cfg: LLMConfig, movie_ids=None, k: int = 5) -> dict:
    hits, info, timings = engine.search_chunks(question, k=k, movie_ids=movie_ids)
    out = {"hits": hits, "info": info, "timings": timings, "answer": None, "error": None, "bad_citations": []}
    if cfg.backend == "none" or not hits:
        return out
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c, _ in hits)
    t = time.perf_counter()
    try:
        text = call_llm(PROMPT.format(context=context, question=question), cfg)
    except requests.exceptions.ConnectionError:
        out["error"] = f"Can't reach the {cfg.backend} server at {cfg.url}."
        return out
    except Exception as e:  # noqa: BLE001 - surface any backend error in the UI
        out["error"] = f"LLM error: {e}"
        return out
    timings["llm_ms"] = 1000 * (time.perf_counter() - t)
    cited = set(re.findall(r"\[(\d+#\d+)\]", text))
    out["answer"] = text
    out["bad_citations"] = sorted(cited - {c["chunk_id"] for c, _ in hits})
    return out


# --------------------------------------------------------------------------- monitoring
LOG_PATH = Path("logs/queries.jsonl")


def log_event(event: dict, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)

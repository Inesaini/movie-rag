"""Build the search index from the Kaggle 'Wikipedia Movie Plots' CSV.

    python build_index.py --csv data/wiki_movie_plots_deduped.csv --n-movies 10000

Writes index/movies.pkl, index/chunks.json, index/embeddings.npy, index/meta.json and
index/validation_report.csv. Re-running only embeds chunks that aren't in the existing index.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rag_core import DEFAULT_MODEL, build_chunks, embed_passages, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/wiki_movie_plots_deduped.csv")
    ap.add_argument("--out", default="index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-movies", type=int, default=10_000, help="0 = all films")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    t0 = time.time()
    raw = pd.read_csv(args.csv)
    df = validate(raw)
    df.drop(columns=["plot"]).to_csv(out / "validation_report.csv", index=False)
    print(df["status"].value_counts().to_string())
    movies = df[~df.status.str.startswith("drop")].reset_index(drop=True)
    movies["movie_id"] = movies.index
    if args.n_movies and len(movies) > args.n_movies:
        movies = movies.sample(args.n_movies, random_state=args.seed).sort_values("movie_id")
    keep = ["movie_id", "title", "year", "origin", "director", "cast", "genre", "url", "plot", "quality", "status"]
    movies = movies[[c for c in keep if c in movies.columns]]
    chunks = build_chunks(movies)
    print(f"{len(movies):,} films -> {len(chunks):,} chunks  (validation {time.time() - t0:.1f}s)")

    # incremental: reuse embeddings for chunks whose text and model haven't changed
    old = {}
    meta_path = out / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["model"] == args.model:
        old_chunks = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
        old_emb = np.load(out / "embeddings.npy")
        old = {c["text"]: old_emb[i] for i, c in enumerate(old_chunks)}
    todo = [i for i, c in enumerate(chunks) if c["text"] not in old]
    print(f"reusing {len(chunks) - len(todo):,} embeddings, embedding {len(todo):,} new chunks with {args.model}")

    t1 = time.time()
    dim = None
    new_emb = embed_passages([chunks[i]["text"] for i in todo], args.model, args.batch_size) if todo else None
    if new_emb is not None:
        dim = new_emb.shape[1]
    else:
        dim = next(iter(old.values())).shape[0]
    emb = np.zeros((len(chunks), dim), dtype=np.float32)
    new_iter = iter(new_emb) if new_emb is not None else iter(())
    todo_set = set(todo)
    for i, c in enumerate(chunks):
        emb[i] = next(new_iter) if i in todo_set else old[c["text"]]
    dt = time.time() - t1

    movies.to_pickle(out / "movies.pkl")
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(out / "embeddings.npy", emb)
    meta_path.write_text(json.dumps({"model": args.model, "n_movies": len(movies), "n_chunks": len(chunks),
                                     "dim": dim, "embed_seconds": round(dt, 1),
                                     "chunks_per_second": round(len(todo) / dt, 1) if todo and dt else None,
                                     "built": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()

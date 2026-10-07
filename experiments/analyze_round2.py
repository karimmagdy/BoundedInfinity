"""Aggregate round-2 results: per-cell mean ± s.e.m. and paired seed comparisons."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from scipy import stats

D = Path(sys.argv[1])


def load(exp):
    recs = []
    for p in sorted(D.glob(f"{exp}.*.jsonl")):
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in recs if r.get("valid")]


def cells(recs, keyf):
    out = defaultdict(dict)
    for r in recs:
        out[keyf(r)][r["seed"]] = r
    return out


def ms(xs):
    return (st.mean(xs), st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def fmt(xs):
    m, s = ms(xs)
    return f"{m:.3f} ± {s:.3f} (n={len(xs)})"


def table(recs, keyf, title):
    print(f"\n## {title}")
    c = cells(recs, keyf)
    for k in sorted(c, key=str):
        rs = list(c[k].values())
        sem = [r["quality"]["semantic_reconstruction_quality"] for r in rs]
        ne = [r["quality"]["nonempty_success_rate"] for r in rs]
        qs = [r["quality"]["query_success_rate"] for r in rs]
        print(f"  {str(k):45s} sem {fmt(sem)}  nonempty {ms(ne)[0]:.3f}  qsucc {ms(qs)[0]:.3f}")
    return c


def paired(c, a, b, label, metric="semantic_reconstruction_quality"):
    if a not in c or b not in c:
        print(f"  {label}: missing cell"); return
    seeds = sorted(set(c[a]) & set(c[b]))
    xa = [c[a][s]["quality"][metric] for s in seeds]
    xb = [c[b][s]["quality"][metric] for s in seeds]
    d = [x - y for x, y in zip(xa, xb)]
    wins = sum(x > 0 for x in d)
    t = stats.ttest_rel(xa, xb) if len(d) > 1 and any(d) else None
    w = stats.wilcoxon(xa, xb) if len(d) > 1 and any(d) else None
    print(f"  {label}: diff {st.mean(d):+.3f} ± {ms(d)[1]:.3f}, A wins {wins}/{len(d)} seeds"
          + (f", paired t p={t.pvalue:.3f}, Wilcoxon p={w.pvalue:.3f}" if t else ""))


sweep = load("sweep")
c = table(sweep, lambda r: (r["backend"], r["cache_size"]), "Sweep (bfs)")
print("\n### bic-aw (A) vs lru-summary-aw-pin (B)")
for k in (8, 16, 32):
    paired(c, ("bic-aw", k), ("lru-summary-aw-pin", k), f"cache {k} semantic")
    paired(c, ("bic-aw", k), ("lru-summary-aw-pin", k), f"cache {k} nonempty", "nonempty_success_rate")
print("\n### bic-aw (A) vs bic (B)")
for k in (8, 16, 32):
    paired(c, ("bic-aw", k), ("bic", k), f"cache {k}")
print("\n### bic-aw (A) vs lru (B)")
for k in (8, 16, 32):
    paired(c, ("bic-aw", k), ("lru", k), f"cache {k}")

hashr = load("hash")
ch = table(hashr, lambda r: (r["backend"], r["cache_size"]), "Hash ablation")
print("\n### bic-hash (A) vs sweep bic-aw (B)")
for k in (8, 16, 32):
    both = {("bic-hash", k): ch.get(("bic-hash", k), {}), ("bic-aw", k): c.get(("bic-aw", k), {})}
    paired(both, ("bic-hash", k), ("bic-aw", k), f"cache {k}")

order = load("order")
co = table(order, lambda r: (r["backend"], r["cache_size"], r["access_order"]), "Access order (cache 16)")
print("\n### bic-aw (A) vs lru-summary-aw-pin (B) by order")
for o in ("bfs", "dfs", "random"):
    paired(co, ("bic-aw", 16, o), ("lru-summary-aw-pin", 16, o), o)
    paired(co, ("bic-aw", 16, o), ("lru-summary-aw-pin", 16, o), o + " nonempty", "nonempty_success_rate")

pilot = load("pilot")
cp = table(pilot, lambda r: (r["backend"], r["cache_size"]), "Coding pilot")
for k in sorted({r["cache_size"] for r in pilot if r["cache_size"]}):
    paired(cp, ("bic-aw", k), ("lru-summary-aw-pin", k), f"pilot cache {k}")

allr = sweep + hashr + order + pilot
print("\nLLM config:", {json.dumps(r["llm"], sort_keys=True) for r in allr} if len({json.dumps(r['llm'], sort_keys=True) for r in allr}) < 3 else "varies")
print("errors:", sum(r["llm_usage"].get("errors", 0) for r in allr),
      "empty_gen:", sum(r["task"].get("empty_generations", 0) for r in allr),
      "runs:", len(allr))

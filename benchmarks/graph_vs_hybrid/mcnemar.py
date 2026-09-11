"""Paired McNemar (exact binomial) on the HotpotQA retrieval matrix."""

import json
from math import comb

RESULTS = "benchmarks/.runs/hotpotqa/results/retrieval_{}.json"


def load(strategy):
    payload = json.load(open(RESULTS.format(strategy)))
    return {r["query_id"]: r for r in payload["queries"]}


def outcome(record, metric):
    """True/False for the metric, or None when the question does not define it."""
    if metric == "hit_at_1":
        return record["rank"] == 1
    return record.get(metric) if metric in record else None


def exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a_name, b_name, metric):
    a, b = load(a_name), load(b_name)
    gained = lost = both = neither = 0
    for qid in a:
        if qid not in b:
            continue
        x, y = outcome(a[qid], metric), outcome(b[qid], metric)
        if x is None or y is None:
            continue
        if x and not y:
            gained += 1
        elif y and not x:
            lost += 1
        elif x:
            both += 1
        else:
            neither += 1
    n = gained + lost + both + neither
    delta = (gained - lost) / n if n else 0.0
    return gained, lost, n, delta, exact_two_sided(gained, lost)


COMPARISONS = [
    ("graph+dense", "dense"),
    ("graph+lexical", "lexical"),
    ("graph+hybrid", "hybrid"),
    ("graph+hybrid", "sparse"),
    ("graph+dense", "graph"),
]

for metric in ("all_at_10", "all_at_2", "hop2_at_10", "hit_at_1", "comparison_at_10"):
    print(f"\n=== {metric} ===")
    print(f"{'comparison':32}{'delta':>9}{'gain/loss':>12}{'n':>7}{'p':>12}")
    for left, right in COMPARISONS:
        gained, lost, n, d, p = compare(left, right, metric)
        discordants = f"{gained}/{lost}"
        print(f"{left + ' vs ' + right:32}{d:+9.3f}{discordants:>12}{n:>7}{p:>12.2g}")

print("\n=== the control: graph alone, bridge vs comparison ===")
for metric in ("comparison_at_10", "bridge_at_10", "hop2_at_10"):
    for rival in ("hybrid", "sparse"):
        gained, lost, n, d, p = compare("graph", rival, metric)
        print(
            f"graph vs {rival:8} {metric:18}{d:+9.3f}  {gained}/{lost}  n={n}  p={p:.2g}"
        )

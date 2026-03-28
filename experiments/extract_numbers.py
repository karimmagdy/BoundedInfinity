#!/usr/bin/env python3
"""Extract experiment numbers for paper tables."""
import json
import os

base = os.path.join(os.path.dirname(__file__), "results")

with open(os.path.join(base, "experiment_2_quality.json")) as f:
    q = json.load(f)
print("=== QUALITY (Experiment 2) ===")
for r in q:
    ql = r['quality']
    mem = r['memory']
    t = r['task']
    lat = r['latency']
    print(f"  {r['backend']:12s} cache={mem['final_size']:4d}  recon={ql['reconstruction_quality']:.3f}"
          f"  success={ql['query_success_rate']:.3f}  mem={mem['peak_bytes']/1024:.0f}KB"
          f"  agents={t['total_agents']}  hit={ql['cache_hit_rate']:.3f}"
          f"  pres={ql.get('info_preservation_ratio', 0):.3f}")

print()
with open(os.path.join(base, "experiment_3_ablation.json")) as f:
    a = json.load(f)
print("=== ABLATION (Experiment 3) ===")
for r in a:
    ql = r['quality']
    print(f"  {r['backend']:20s} recon={ql['reconstruction_quality']:.3f}"
          f"  success={ql['query_success_rate']:.3f}"
          f"  evictions={r.get('evictions', '?')}"
          f"  pres={ql.get('info_preservation_ratio', 0):.3f}")

print()
with open(os.path.join(base, "experiment_4_stress.json")) as f:
    s = json.load(f)
print("=== STRESS (Experiment 4) ===")
for r in s:
    ql = r['quality']
    mem = r['memory']
    t = r['task']
    print(f"  {r['backend']:12s}  agents={t['total_agents']:5d}  recon={ql['reconstruction_quality']:.3f}"
          f"  success={ql['query_success_rate']:.3f}  mem={mem['peak_bytes']/1024:.0f}KB"
          f"  hit={ql['cache_hit_rate']:.3f}")

print()
with open(os.path.join(base, "experiment_1_scale.json")) as f:
    sc = json.load(f)
print("=== SCALE (Experiment 1, depth=3) ===")
for r in sorted(sc, key=lambda x: (x['memory']['final_size'], x['backend'])):
    if r['task']['tree_depth'] == 3 or r['config']['max_depth'] == 3:
        mem = r['memory']
        ql = r['quality']
        t = r['task']
        print(f"  {r['backend']:12s} cache={mem['final_size']:4d}  agents={t['total_agents']:5d}"
              f"  recon={ql['reconstruction_quality']:.3f}  success={ql['query_success_rate']:.3f}"
              f"  mem={mem['peak_bytes']/1024:.0f}KB")

Synthetic experiments 1-4 re-run on 6 Oct 2026 with both evaluation fixes:
the 2 Oct reference/empty-score fix and the stale `_reconstructed` fix (commit e281bad).

- experiment_{1..4}_*.json: both fixes (use these).
- prefix_only_reference_fix_*.json: 2 Oct fix only (pre-e281bad runtime), kept to
  attribute each change for the response letter.
- The paper-era files in ../experiment_*.json predate both fixes, and do not exactly
  match the numbers printed in main.tex either.

Memory caveat: `memory.peak_bytes` (tracemalloc) spans the swarm run AND the
measurement loop that queries every agent afterwards. For the 3,280-agent stress test,
the swarm-run peak alone is 19.6 MB (BIC) vs 17.9 MB (LRU); the measurement loop drives
BIC's recorded peak much higher. Report the swarm-run peak for working-state memory.

experiment_4_stress.json was re-run after the harness change and carries
`memory.run_peak_bytes` (swarm run only): BIC 19,195 KB, cheapest (FIFO) 17,385 KB,
ratio 1.10. About 17 MB of every method's figure is the runner's own record of the
3,280 synthetic answers, common to all methods.

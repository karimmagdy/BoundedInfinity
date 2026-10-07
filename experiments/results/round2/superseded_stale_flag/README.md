Round-2 records as first run (6 Oct 2026), before the stale `_reconstructed` fix
(commit e281bad). The bic / bic-aw / bic-hash quality numbers here are WRONG:
agents evicted before they ran kept the flag after execution and were scored as
empty reconstructions while still cached. The LRU, lru-summary-aw-pin and
unbounded records are unaffected and were kept in the live files; only the BIC
arms were re-run with the fix.

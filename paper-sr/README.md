# BoundedInfinity_SR — Nature Scientific Reports submission package

This directory contains the Scientific Reports (SR) journal-version
draft of the *Bounded Infinity Cache* manuscript. The original
NeurIPS 2026 / TMLR version remains untouched in
`../../BoundedInfinity/` and must not be edited from this directory.

**Latest revision (2026-05-03)**:
1. Reviewer-weakness pass — 9 fixes (W1–W10).
2. Tier A acceptance-push pass — 6 additional text-only fixes (A1 concrete I, A2 worst-case ρ, A3 exhaustive 41-byte breakdown, A4 M≥d+1 in-practice, A5 concurrency model, A6 OPENDEV comparison).
3. **Tier B acceptance-gate experiments (B1, B2, B3) — EXECUTED.** New `LRUSummaryAWBackend` and `LRUPinBackend` in `experiments/baselines.py` (reviewer Q1 and Q5 strengthened-baseline asks). New `experiments/run_llm_coding_pilot.py` for multi-agent coding pilot (reviewer Q8 additional-workload ask). Two new tables in `main.tex`: `tab:tierb_baselines` (closure-of-gap analysis showing LRU+Pin closes only 29% of the BIC gap; LRU+Summary-AW only 12%) and `tab:coding_pilot` (BIC's 100% query success holds on a fresh task family). The "Future case study (commitment)" paragraph has been replaced by the actual coding-pilot results.

Both reviewer-stated acceptance gates from the Overall Assessment are now empirically addressed. See `SUBMISSION_NOTES.md` §0 for full Tier B numbers and methodology.

## 1. Files in this directory

| File | Purpose |
|------|---------|
| `main.tex` | Complete SR-format manuscript. Single-column article, `\documentclass[fleqn,10pt]{article}`, Nature end-matter order. Uses `\input`-free structure (one file, easier for SR submission). |
| `references.bib` | Bibliography — the NeurIPS references plus 2024–2025 agent-memory citations (MemAgents workshop, A-MEM, memory surveys, etc.). |
| `SUBMISSION_NOTES.md` | SR submission metadata: suggested subject areas, draft cover letter, suggested reviewers, data / code availability / competing interests / author contribution statements, and a mapping from SPRO / REVISION_PLAN issues to the fixes in this draft. |
| `README.md` | This file — index and diff notes vs the NeurIPS version. |

## 2. Compiling

The manuscript references figures from the NeurIPS source tree using
relative paths:

```
../../BoundedInfinity/docs/paper/figures/quality_vs_cache.pdf
../../BoundedInfinity/docs/paper/figures/stress_test.pdf
```

To compile from this directory:

```bash
cd /Users/kmagdy-ma-eg/Workspace/Research/ScientificReports_Submissions/BoundedInfinity_SR
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Packages required: standard TeX Live distribution (2024+) with
`natbib`, `hyperref`, `algorithm`, `algorithmic`, `authblk`,
`pifont`, `booktabs`, `multirow`, `microtype`, `setspace`,
`mathtools`. All standard on MacTeX / TeX Live.

For SR submission (which typically asks for a single PDF plus the
source bundle), it may be convenient to inline the figures into this
directory — see section 5 below.

## 3. What's the same as the NeurIPS version

- Core architecture (Cantor pairing, fixed-size slot array,
  hierarchical summarization, optional Hilbert index).
- All four theorems and their proofs.
- The formal definitions of the agent swarm, BIC, and information
  preservation ratio.
- The main experimental numbers (Tables 1–4, Figures 1–2).
- Ablation conclusion: hierarchical summarization is the critical
  component; Hilbert is not a demonstrated contribution in the
  current experimental settings.

## 4. What's different from the NeurIPS version (diff notes)

The SR draft is a *journal version* of the NeurIPS submission. The
goal is a more accessible framing for Nature's multidisciplinary
audience and a cleaner treatment of known weaknesses. Concretely:

### Structural changes

- **Section order** follows Nature's Research Article template:
  Abstract → Introduction → Results → Discussion → Methods →
  References → Acknowledgements → Author Contributions →
  Competing Interests → Data Availability → Code Availability.
  The NeurIPS version used: Abstract → Introduction → Related Work
  → Method → Experiments → Discussion → References → Appendix Proofs.
- **Proofs moved into Methods.** The NeurIPS appendix-proofs are
  inlined into the Methods section's "Theorems" subsection. Each
  proof is a few sentences with the mechanical details compressed;
  the structure of the argument is fully preserved.
- **No separate Related Work section.** The SR style absorbs related
  work into the Introduction. The 2024–2025 agent-memory references
  (MemAgents, A-MEM, memory surveys) are woven into the motivating
  narrative rather than listed in a standalone section.
- **Hilbert extension moved from Appendix B to Methods.** Presented
  as an optional architectural provision inside Methods with a clean
  locality-bound lemma, rather than a separate appendix block. The
  honesty framing ("Hilbert is not a demonstrated contribution") is
  preserved verbatim.
- **NeurIPS Checklist removed.** Not applicable to SR.

### Framing changes (per SPRO / REVISION_PLAN / 2026-04-27 advisor pass)

- **Title (final, 2026-04-27).** Now
  "Bounded Infinity Cache: Memory-Bounded State Management
  for Recursive LLM Agent Swarms" --- 11 words, preserves the
  named-system anchor ("Bounded Infinity Cache"), drops the
  "Bounding Unbounded" wordplay, and explicitly signals
  *memory management / cache system* for *multi-agent LLM*. (An
  intermediate working title used the phrase
  "cache-bounded state management for unbounded multi-agent large
  language model swarms"; the advisor pass replaced it with
  the system-anchored form.)
- **Abstract leads with 4.1× real-LLM number**, per REVISION_PLAN
  W-Abs-1. The 9.4× synthetic number is still present but
  demoted to a supporting confirmation sentence.
- **Registry O(N) growth is explicit from Introduction onward.** The
  NeurIPS version treats it as a Limitation in Discussion; the SR
  version states it in the Introduction, in the abstract
  ("cache-bounded"), in the main theorem, and again in Discussion.
- **Accessible glossary in Introduction.** "LLM agent", "Cantor
  pairing", "Hilbert curves", and "eviction" each get a one-sentence
  definition in plain prose. A worked sticky-note intuition example
  is placed before the formal model, per the SR-audience requirement.
- **Wall-clock timing in Results.** The NeurIPS version had this in
  compute-resources checklist answers; SR places it in the Results
  section as a dedicated paragraph.
- **IPR theory-vs-empirical gap** has its own Discussion paragraph —
  explains the infimum-vs-average distinction clearly and flags a
  tighter Monte-Carlo estimator as future work.
- **LRU+Summary baseline description** is fully spelled out in
  Methods, including the parent-write / drop-if-parent-evicted
  behaviour, addressing REVISION_PLAN W-Exp-3.
- **Architecture narrative** expanded from the NeurIPS caption-level
  description to a five-step data-flow narration in Methods
  (addressing P2.5).

### Bibliography additions

Initial 2024-2025 set (from earlier passes):

- `memagents2026` — ICLR 2026 MemAgents workshop.
- `amem2025` — A-MEM (arXiv:2502.12110).
- `zhang2024surveymemory` — 2024 agent memory survey.
- `zhong2024memorybank` — MemoryBank (AAAI 2024).
- `hu2024memwalker` — MemWalker.
- `wang2024autoagents` — AutoAgents.
- `wang2024voyager` — Voyager (TMLR 2024).
- `xi2024agentsurvey` — 2025 LLM-agent survey.
- `belady1966study` — Belady's classical page-replacement study
  (used to anchor the OS analogy).
- `denning1970working` — Denning's working-set model.
- `tanenbaum2014modernos` — standard OS reference.
- `cantor1878beitrag` — Cantor's original pairing paper.
- `butz1971hilbert` — renamed from `butz1969hilbert` in the NeurIPS
  source (the paper is actually 1971; SPRO Appendix D / REVISION_PLAN
  P2.6 minor fix).

Added in the 2026-04-27 advisor revision pass (Items 3 and 5):

- `vitter1985reservoir` — Vitter's reservoir-sampling paper (TOMS 1985).
- `metwally2005spacesaving` — Metwally et al., space-saving frequency
  estimator (ICDT 2005).
- `zhang2020bertscore` — BERTScore (Zhang et al., ICLR 2020); cited
  in the new "Stronger semantic metrics" paragraph.
- `reimers2019sbert` — Sentence-BERT (Reimers & Gurevych, EMNLP 2019);
  cited for `all-mpnet-base-v2`-style sentence embeddings.
- `xiao2024bge` — BGE / C-Pack (Xiao et al. 2024); cited for the
  `bge-large-en-v1.5` checkpoint.

### 2026-04-28 follow-up advisor pass changes (Items A, B, C)

Three further advisor items applied to `main.tex` and `references.bib`:

- **Item A — Cost analysis subsection** (Results, lines 418--510):
  new `\subsection*{Cost analysis: memory vs.\ compute}` after the
  Wall-clock paragraph, with a `tab:cost_analysis` table reporting
  per-attempted-query and per-successful-query unit costs (BIC at
  $\slots{=}8$ is 5.24× cheaper than LRU per successful query;
  2.62× at $\slots{=}16$; 1.36× at $\slots{=}32$).
- **Item B — Failure modes subsection** (Discussion, lines
  925--1005): new `\subsection*{Failure modes}` covering small-cache
  ($\slots \leq 4$), deep trees ($d \geq 5$), heterogeneous
  siblings, and near-duplicate sibling states, each with a
  one-paragraph mitigation. Carries a `% TODO: Karim, add concrete
  trace if available` for an optional log-excerpt insertion.
- **Item C — Case study: research-assistant pipeline** (Results,
  lines 512--589): new `\subsection*{Case study: research-assistant
  pipeline}` immediately after Cost analysis. Reframes the existing
  50-agent Gemini~2.0~Flash experiment as a research-assistant
  use case (no new compute), casts query success as task coherence,
  states the 100\% (BIC) vs 19\% (LRU) headline coherence claim and
  the ~5× research-output-per-dollar figure, and commits to a
  follow-up multi-agent coding case study (LangGraph-style recursive
  code generation with unit-test feedback).

New BibTeX entry: `gemini2026pricing` (Google Gemini API pricing
page, accessed 2026-04). Headline framing throughout the new
content uses 4.1× consistently — the older 4.1× framing has not
been reintroduced.

### 2026-04-27 advisor pass changes (summary)

Five advisor items applied directly to `main.tex` and `references.bib`:

1. **Title** changed to *Bounded Infinity Cache: Memory-Bounded
   State Management for Recursive LLM Agent Swarms* (memory /
   cache framing for multi-agent LLM systems, system-anchored).
2. **Abstract** rewritten in 162 words: opens with the practical
   problem, states the BIC contribution, foregrounds the 4.1×
   real-LLM number, mentions the 3,280-agent stress test, closes
   with the targeted "concrete memory-management primitive for
   recursive LLM agent systems" line.
3. **Overclaim sweep + new related-work subsection.** "First" /
   "only" claims softened. New unnumbered subsection
   "Relation to memory-management traditions" added in the
   Introduction (`main.tex` lines 157--210), with three lineage
   blocks: OS memory management, streaming/sketching, RAG /
   LLM-memory. Five new BibTeX entries added (see above).
   Contributions list reframed to present BIC as a specific
   recombination.
4. **Algorithm 1/2 mechanism annotations.** Each step in
   `\textsc{Spawn}` and `\textsc{Query}` now carries a
   `\COMMENT{...}` flagging the implementation mechanism (Cantor
   pairing → closed-form arithmetic; Hilbert curve → closed-form
   bit-interleaving; eviction summarization → LLM-based, Gemini 2.0
   Flash; spawn → heuristic). A new
   `\paragraph{Summarization prompt.}` block in Methods gives the
   verbatim ≤200-token prompt template used at eviction time.
5. **Stronger semantic metrics** section in Discussion. Two new
   `\paragraph{...}` blocks: "Stronger semantic metrics" (honest
   acknowledgement of Jaccard / TF-IDF cosine as surface-level
   lexical proxies) and "Stronger metrics (planned protocol)"
   (drop-in replacement protocol using sentence-embedding cosine,
   BERTScore, and LLM-as-judge scoring). The Limitations list
   expanded to five items; the Methods Metrics paragraph now
   cross-references the limitation honestly.

### Known issues NOT resolved in this draft

The following are flagged in `SUBMISSION_NOTES.md` as open questions
for Karim rather than fixed in the draft:

- **Second LLM provider experiment** (GPT-4o-mini / Claude 3.5 Haiku)
  — REVISION_PLAN P3.5.
- **200-agent LLM experiment** for rebuttal prep — REVISION_PLAN
  P3.1 / SPRO Appendix E Q2.
- **Registry-eviction prototype** — Discussion Limitation 1 remains
  an open limitation; a disk-paged supplementary experiment would
  resolve it.
- **Final GitHub repo URL** — currently a placeholder
  (`https://github.com/karimmagdy/BoundedInfinity`).

## 5. Submission checklist (before uploading to SR portal)

- [ ] Replace the placeholder GitHub URL in Code Availability with the
      final public URL / Zenodo DOI.
- [ ] Verify the Gemini 2.0 Flash numbers in Table 1 are final — if
      a later run is available, update all 12 cells.
- [ ] Confirm author affiliations and corresponding-author email.
- [ ] Optionally: copy the two PDF figures from
      `../../BoundedInfinity/docs/paper/figures/` into this directory
      and change the `\includegraphics` paths from
      `../../BoundedInfinity/docs/paper/figures/FILE.pdf` to
      `FILE.pdf`, so the SR submission is a self-contained bundle.
- [ ] Compile one final pass; verify no `LaTeX Warning: Citation ...
      undefined` and no `Reference ... undefined` messages.
- [ ] Verify the compiled PDF's figure references (Figure 1, Figure 2)
      and table references (Tables 1–4) all resolve.
- [ ] Upload cover letter from `SUBMISSION_NOTES.md` section 2.
- [ ] Select subject areas from `SUBMISSION_NOTES.md` section 1.
- [ ] Provide suggested reviewers from `SUBMISSION_NOTES.md`
      section 3.

## 6. Do not edit upstream

Per explicit user instruction: no file under
`/Users/kmagdy-ma-eg/Workspace/Research/BoundedInfinity/` is modified
by this SR package. The SR draft lives entirely under this directory
and references the upstream figures read-only.

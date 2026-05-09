# Cover Letter — BoundedInfinity

**Paste this into the SR portal "Cover Letter" field.**

---

Dear Editors of *Scientific Reports*,

We are submitting the manuscript "**Bounded Infinity Cache: Memory-Bounded State Management for Recursive LLM Agent Swarms**" for consideration as a research article.

Multi-agent systems built on large language models — frameworks such as LangGraph, AutoGen, CrewAI, and MetaGPT — store each spawned agent's state in unbounded data structures. Long-running swarms therefore exhaust host memory or must be truncated by hand, a fundamental scalability barrier. We introduce the **Bounded Infinity Cache (BIC)**, a fixed-size slot array addressed by the Cantor pairing function, equipped with hierarchical summarization that folds evicted state into ancestors at eviction time. We prove four guarantees for this design: cache-bounded memory, zero external fragmentation, universal retrievability with O(log N) miss cost, and indefinite liveness; we are also transparent that a lightweight registry still grows as O(N), a limitation we discuss in the paper.

The headline empirical result is a real **50-agent swarm driven by Gemini 2.0 Flash**, run across three cache capacities and five random seeds: BIC delivers **4.1× higher effective semantic reconstruction quality** than LRU at 20% retention while maintaining **100% query success vs. 19% for LRU**. A 3,280-agent synthetic stress test reproduces the effect at scale. An ablation study clarifies the architecture: hierarchical summarization is BIC's load-bearing mechanism, and the deterministic Cantor addressing supplies the structural target into which summaries land — both ingredients are required (LRU+Summary alone collapses to bare LRU on the stress test).

We believe *Scientific Reports* is the right home because the contribution is formally rigorous yet experimentally grounded, and is likely to be of interest across computer science, machine learning, and applied mathematics — a multidisciplinary profile well matched to SR's editorial scope. The code is open-source under Apache 2.0 and all experiments are reproducible from the attached repository.

We declare no competing interests. The manuscript is not under consideration elsewhere.

Sincerely,
Karim Magdy
(On behalf of all authors)

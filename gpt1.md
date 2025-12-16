You’re in a great spot: the “dream → codegen → train → benchmark → feedback” loop is basically closed, you’re logging insights, you’ve got reproducible recipes, and you’ve already found real wins (DeepGQA/MQA18, MQAMamba, etc.). So the next steps are mostly **(1) harden + automate**, then **(2) push the search frontier**.

## Do these next (highest ROI)

### 1) Unblock paper ingestion DX (open: `arcfusion-2si`)

Right now ingestion is brittle because it depends on `ANTHROPIC_API_KEY` being in the shell.

* Add `.env` support (python-dotenv) and a clear fallback error message.
* Update README/CLAUDE.md with the canonical workflow.
  This pays off immediately because it reduces friction every time you expand the component DB.

### 2) Fix the “compaction during training” issue (open: `arcfusion-ekl`)

This is a silent productivity killer.

* Make long training runs start from a “fresh context” path (separate command/script entrypoint).
* Optionally add a “preflight compaction” step in whatever runner you’re using so it never happens mid-run.

### 3) Lock in “results → insights → dreaming” as the default loop

You already implemented training_insights + results-aware composition. Now operationalize it:

* After every batch/grid run, auto-generate:

  * 1–3 “top insights” (already supported)
  * 5–10 “next hypotheses” (new: based on gaps/pareto front)
* Make this a single command you can run daily:

  * `arcfusion leaderboard`
  * `arcfusion insights`
  * `arcfusion summary recipe-cards`

## Then push forward on research

### 4) Go after Mamba speedups (open: `arcfusion-8ys`)

You’ve proven quality; now buy back time.

* Start with the simplest win first: kernels / fused ops / better implementations (whatever fits your stack).
* Measure with the same fair harness so you can attribute speed changes cleanly.

### 5) Decide “compute provider strategy” (open: `arcfusion-gb6`)

You already moved to A100 and saw gains. Next step is to choose whether Modal stays the default or becomes “one backend.”

* Make a tiny bake-off harness: same model/config, run on 2 providers, compare **time-to-first-token**, **throughput**, **failure rate**, and **cost-per-run**.
* Outcome: a single recommended “default provider” + a fallback.

## Optional but powerful (if you want to scale contributors)

### 6) Multi-agent autonomous development system (open: `arcfusion-a8o`)

This is worth doing *only after* the repo has:

* a rock-solid “local validate” step (you have this),
* reliable CI,
* clear contribution boundaries (labels like `good-first-issue`, `agent-safe`, etc.).

If you do it now, start super small: **one “worker” that only touches docs/tests**, not core training code.

## A concrete 3-session plan (no fluff)

**Session 1 (stability):**

* Close `arcfusion-2si` (.env + docs)
* Close `arcfusion-ekl` (separate runner / preflight)

**Session 2 (automation):**

* Add a “daily loop” command or script that prints:

  * current pareto winners,
  * top 5 insights,
  * 5 suggested next experiments (generated from gaps)

**Session 3 (performance):**

* Start `arcfusion-8ys` with one measurable speed improvement and benchmark it against your current best quality model + best efficiency model.

If you tell me what you want the project to optimize for *right now* (pure quality, best quality/time, or “most likely to become a real usable small LM”), I’ll translate that into a very specific next experiment grid (architectures + configs + stopping criteria) that matches your current findings (DeepGQA depth scaling, MQA18, MQAMamba sweet spot, etc.).

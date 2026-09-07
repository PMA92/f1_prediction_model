# Roadmap: F1 model → portfolio-grade ML project

Target: 3 resume bullets, in order of what they prove —
1. Rigorous, leak-free data/eval infrastructure
2. Honest baseline comparison + measured ceiling
3. Calibrated probabilistic outputs (not point predictions)

Every number below is measured (walk-forward backtest, 2024-2025 data unless
noted), not estimated. Re-verify after Phase 1 extends the dataset to
2022-2025 — the numbers may shift.

---

## Phase 0 — Repo hygiene (do first, low effort, unblocks everything)

- [x] Reconcile machines: Windows clone pulled forward to `origin/master` (2026-09-07)
- [ ] Resolve the stashed WIP (`stash@{0}`, "Teleport auto-stash") — it's a
      broken, superseded sklearn feature-engineering attempt. Recommend
      dropping it once confirmed unneeded: `git stash drop`
- [ ] Delete or archive `build_data.ipynb` cells 6-7 (the `LabelEncoder` /
      `StandardScaler` / `GradientBoostingRegressor` tail). It errors on
      execution and reflects an abandoned approach — `test.py`'s
      `DictVectorizer` pattern replaced it. A reviewer reading this repo
      should not find dead, broken code.
- [ ] Add `requirements.txt` (pandas, numpy, scikit-learn, scipy, fastf1,
      openpyxl, pytest) with pinned versions — currently nowhere in the repo
- [ ] `pip install pytest` on any machine used for this project

## Phase 1 — Data pipeline: finish what's collected

- [ ] Fix the one remaining bug in `predictions2.ipynb::build_year_data`:
      guard `pd.concat(frames)` when `frames` is empty
      (`if not frames: print(...); return`)
- [ ] Run `build_year_data` for 2022-2025 on this machine →
      `data_{2022..2025}.xlsx` + `weather_{2022..2025}.xlsx`
      (expect ~1,838 driver-races over 92 races; API returns 429s — the
      per-round `try/except` means a throttled race is skipped and logged,
      not a lost season; re-run any year with `SKIPPED` lines)
- [ ] Decide fate of FP2 practice-pace collector (`scratchpad/pace_collector.py`
      from prior session) — deferred by decision on 2026-08-16 (+0.027 rho,
      not worth a second collection pass yet). Revisit only after Phase 4.

## Phase 2 — Restructure into an importable package

Notebooks are fine for exploration; a work sample needs modules and tests.

- [ ] `src/f1/collect.py` — the `build_year_data` logic, notebook-independent
- [ ] `src/f1/features.py` — feature construction, with an explicit
      `KNOWN_BEFORE_LIGHTS_OUT` constant (pre-race + rolling-with-shift(1)
      features only) kept separate from anything measured during the race
      being predicted
- [ ] `src/f1/evaluate.py` — restore the walk-forward harness. `test.py` is
      currently 0 bytes; the working version (train-only `DictVectorizer`
      fit, `backtest_year` walk-forward, grouping by `['Year','Round']` not
      `EventName` since track names repeat across seasons) existed earlier
      this session and should be rebuilt from that design, not from scratch
- [ ] `tests/` directory, `pytest` passing on a clean checkout

## Phase 3 — Goal 1: automated leakage test

This is the single most differentiating artifact in the whole project —
almost no portfolio ML project has one.

- [ ] `tests/test_leakage.py`: for a held-out round N, rebuild every rolling
      feature using only rows with `Round < N`, and assert it matches the
      value the production pipeline assigned to round N
- [ ] Add a negative control: deliberately feed the test an in-race column
      (e.g. race-lap-time-derived) and assert the test *fails*. A leakage
      test that can't fail proves nothing.
- [ ] Get it green in `pytest`
- [ ] Stretch: GitHub Actions workflow running `pytest` on push — this is
      what makes "leakage test in CI" a literally true resume claim rather
      than a local script

## Phase 4 — Goal 2: baseline-anchored walk-forward harness

- [ ] Re-run the grid-position-only baseline on the full 2022-2025 dataset.
      Prior measurement (2024-2025 only, 958 rows): **MAE 3.34, Spearman
      ρ +0.651**. Confirm this holds (or shifts) at ~1,838 rows.
- [ ] Refactor `evaluate()` so it structurally cannot report a model metric
      without the baseline printed alongside — not a convention to remember,
      a function signature that enforces it
- [ ] Re-run the six previously-tested feature families (one-hots, rolling
      finish-form, rolling race-pace, teammate-relative pace, race-trim
      delta, FP2 gap) on the bigger dataset and update the numbers. Prior
      result: five of six did not beat baseline; only FP2 long-run pace did
      (ρ 0.629 → 0.671, deferred — see Phase 1)
- [ ] Quantify and document the ceiling: DNFs are ~12% of rows but ~50% of
      positional movement, and are near-unpredictable pre-race
      (walk-forward AUC 0.575 vs 12.5% base rate). This number is the
      strongest evidence of rigor in the whole project — it belongs in the
      README, not buried in a notebook.

## Phase 5 — Goal 3: calibrated probabilistic outputs

Point-prediction accuracy has a low, mostly-DNF-bound ceiling (~ρ 0.76). This
reframe sidesteps that ceiling with a well-posed problem instead.

- [ ] Derive binary targets from `FinishingPosition`: win (P1), podium (≤P3),
      points (≤P10)
- [ ] Train a classifier per target with the same walk-forward discipline as
      the regression harness (refit before each round, no shuffled splits)
- [ ] Score with Brier score and log loss against a naive baseline (e.g.
      probability derived from grid-position rank)
- [ ] Produce a reliability diagram (predicted probability bucket vs actual
      frequency) — this is the chart that proves calibration, not just
      accuracy
- [ ] If raw output is miscalibrated, apply Platt scaling or isotonic
      regression and re-measure
- [ ] Optional, high value if obtainable: compare calibrated probabilities
      against closing betting odds — the strongest baseline that exists,
      and matching or losing to it by a known margin is a serious,
      honestly-reported result either way

## Phase 6 — Package it as a work sample

- [ ] `README.md` telling the actual story: baseline → what was tried →
      what failed and why (with numbers) → measured ceiling → final
      approach → limitations. The negative results are the credibility,
      not something to omit.
- [ ] Final resume bullets re-verified against the Phase 4/5 numbers (not
      the 958-row numbers currently drafted) before this goes in front of
      anyone
- [ ] Model card / limitations section: DNF unpredictability, weather-as-
      outcome vs weather-as-forecast, dataset era (2022-2025 ground-effect
      regs only, not comparable to older seasons)

---

## What's already true and provable today

- Leak identification: `IN_RACE` lap-time features were caught and removed
  from the honest evaluation path
- Train-only encoder fitting (`DictVectorizer.fit` on train, `.transform`
  only on test)
- Walk-forward-by-round backtest design (exists in history, needs restoring)
- A measured, reported case where the trained model lost to a trivial
  baseline (958-row set: model ρ +0.609 vs grid ρ +0.651) — rare intellectual
  honesty, keep this in the README even after it's fixed

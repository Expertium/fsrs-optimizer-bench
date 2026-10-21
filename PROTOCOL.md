# FSRS Optimizer Bench protocol

This is the measurement contract for FSRS Optimizer Bench. A faster optimizer only counts when it remains inside the accuracy band on the fixed user set.

## Frozen benchmark
- **Dataset:** bundled at [`data/revlogs`](https://github.com/Expertium/fsrs-optimizer-bench/tree/main/data/revlogs), with one partition for each user 1ΓÇô50. It is derived from the public [`open-spaced-repetition/anki-revlogs-10k`](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k) dataset and processes 1,897,936 optimizer items.
- **Input mode:** include short-term reviews, use elapsed seconds converted to days, and apply the existing recency weighting (`--short --secs --recency`). Preserve item order, filtering, seed, and batch construction.
- **Model:** finished 34-parameter FSRS-7. Reference equations are in [`models/fsrs_v7.py`](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/models/fsrs_v7.py); the Rust recurrence is in `fsrs-rs/src/model.rs`.
- **Optimizer baseline:** unoptimized Burn/autodiff implementation, 9 epochs, batch size 256, seed 2023, learning rate 0.0282, Adam beta values 0.55 and 0.9942, and per-epoch validation/best-epoch selection. Preserve the exact baseline implementation when evaluating candidates. Do NOT modify hyperparameters (n epochs, learning rate, betas for Adam, etc.) throughout your 20 experiments.
- **Timed region:** only the Rust `compute_parameters()` call, timed with a monotonic clock inside the Python binding. Python input conversion and evaluation are excluded.
- **Accuracy:** after timing, evaluate the trained parameters on the user's same items with the frozen Rust evaluator. The arithmetic mean of the 50 per-user LogLoss values must be **[0.3066, 0.3096] inclusive** (0.3081 ┬▒ 0.0015).
Keep the model and inputs fixed; otherwise the results are not comparable.
## Experiment scope
Each experiment must change one conceptual thing; do not bundle unrelated changes into one run. A major refactor is acceptable when it is a cohesive change to one concept. Record the exact source revision and candidate build used for every result.

## Timing and acceptance
Run each user **three times per build** and use the minimum Rust-call time for that user. Interference only adds time; the minimum is a less biased estimate of the compute floor. The supplied `--paired BASELINE CANDIDATE` harness reports the **median of paired per-user ratios** `baseline_ms / candidate_ms`, not the ratio of the two medians. A speedup is accepted only if this median ratio is at least **1.015** (a 1.5% improvement) and the candidate's full-run mean LogLoss is inside the accuracy band. A completed run that misses either criterion is **rejected** and consumes one of the 20 experiments. A run that fails due to an error before producing a valid result is **not rejected** and does not consume an experiment; you may fix the error and retry under the same experiment number. A result within measurement uncertainty of the threshold should be rerun in a new session before acceptance.
 ## Protected baseline and experiment isolation
Create the local golden baseline once with `scripts/experiment.ps1 -Action prepare-baseline`. It builds the original optimizer from commit `0485116` in a disposable detached worktree, stores the resulting extension and SHA-256 manifest under the ignored top-level `baseline/` folder, and refuses to overwrite an existing artifact. Verify the manifest hash before each use; if the binary is missing or damaged, rebuild it from the pinned commit rather than substituting a candidate.

Create every experiment with `scripts/experiment.ps1 -Action new -Experiment N`. This makes a clean detached Git worktree outside the repository at `../fsrs-optimizer-bench-experiments/run-NN` and places a copy of the golden extension and its manifest there. Work only in that experiment's directory. Each experiment must have its own source tree, Cargo target directory, generated output names, and candidate extension. Do not reuse experiment worktrees or candidate binaries. The canonical checkout and top-level golden baseline remain untouched while experiments run. Use unique run names and preserve the baseline/candidate hashes with results. This keeps changes and build artifacts isolated between experiments.
On supported Windows x86-64 machines, the paired harness runs both builds simultaneously for each user without CPU affinity or priority changes. Other platforms may run the two extensions sequentially per user using the same three timing repeats and minimum-of-three rule; mark these measurements unpaired. For paired A/B timing:
1. Start baseline and candidate measurements for the same user's identical inputs in separate processes at nearly the same time. Do not adjust processor affinity or priority.
2. Alternate which process is launched first for each repetition.
3. Do not advance to the next user until both builds finish the current user's measurements.
4. Use each build's minimum of three timings; calculate per-user `baseline / candidate` ratios and their median.
5. Evaluate both builds after timing; confirm the candidate remains in-band and record the baseline/candidate accuracy difference.
Sequential fallback measurements may be used when simultaneous processes are unavailable; alternate build order, retain the same repeats, and label them unpaired. Do not treat runs from different machines or substantially different load conditions as paired.
### Setup (Windows)
Do not change power plans, process priority, affinity, or processor frequency settings from their normal configuration. Record CPU model, OS, Python/Rust versions, date, process count, and whether timing was paired.
## Leaderboards and agent iteration logs
The model leaderboard in `README.md` is maintained by the human project owner. Agents must not edit or reorder it.

Each agent must maintain an experiment `.md` file in its isolated worktree, with one row for every completed run and these columns:

| iter | median time_before (ms) | median time_after (ms) | speed_ratio | log loss before | log loss after | accepted/rejected | summary of the change |
|---|---:|---:|---:|---:|---:|---|---|

- `iter` is the experiment number (1–20). An error or failed/incomplete run does not consume an iteration; fix it and retry using the same number. It is not marked accepted or rejected.
- `median time_before` and `median time_after` are medians of per-user times in milliseconds, using each user's minimum of three Rust-call timings.
- `speed_ratio` is the **median of per-user ratios** `baseline_ms / candidate_ms`, not the ratio of median times. It must be at least `1.015` to pass.
- `log loss before` and `log loss after` are the baseline and candidate means across the fixed 50 users. The candidate must be in the accuracy band.
- Mark a completed run accepted only when both the speed threshold and accuracy band pass; otherwise mark it rejected.
- `summary of the change` is written by the agent and describes the single conceptual change.

Record model/provider and slug, prompt/version, API cost and available token counts, source revision, and machine/software environment as metadata in the same file. The human updates the README model leaderboard; the agent-authored file is the detailed run log.
## Fairness and implementation constraints
- **CPU only:** GPU execution, GPU-specific libraries, and accelerator offloading are prohibited.
- **Single-user optimization only:** `compute_parameters()` must optimize exactly one user's training set per call. Joint optimization, cross-user batching, and vectorizing work across users (including multi-user SIMD) are prohibited; benchmark throughput improvements must come from a faster per-user optimizer call.
- **Timed work:** all work needed to optimize a user's parameters must remain inside the timed Rust `compute_parameters()` call. Do not precompute, cache, asynchronously perform, or move optimizer work into Python preprocessing, initialization, evaluation, background threads, or any other untimed region. Only deterministic input preparation and post-training evaluation are outside the timed interval.
- **Portable implementation:** CPU only; GPU execution, GPU-specific libraries, accelerator offloading, runtime CPU-specific dispatch, and machine-specific build flags are prohibited. SIMD is permitted only with baseline SSE2 instructions, which are part of the x86-64 baseline; do not require newer instruction sets. Compile for the generic x86-64 baseline, not the build host's native CPU.
- **Reproducibility:** preserve the same fixed input and model semantics, and report any nondeterminism. Do not suppress failed users or change the scoring population.
- **Scope:** keep candidate changes focused on one conceptual change. Do not bundle multiple unrelated changes into a single experiment.

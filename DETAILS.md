# Measurement protocol and correctness requirements
This is the measurement contract for FSRS-Optimizer-Bench. A faster optimizer only counts when it remains inside the accuracy band on the fixed user set.
## Frozen benchmark
- **Dataset:** `anki-revlogs-10k`, users 1-50; the run processes 1,897,936 optimizer items.
- **Input mode:** include short-term reviews, use elapsed seconds converted to days, and apply the existing recency weighting (`--short --secs --recency`). Preserve item order, filtering, seed, and batch construction.
- **Model:** finished 34-parameter FSRS-7. Reference equations are in [`models/fsrs_v7.py`](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/models/fsrs_v7.py); the Rust recurrence is in `fsrs-rs/src/model.rs`.
- **Optimizer baseline:** unoptimized Burn/autodiff implementation, 9 epochs, batch size 256, seed 2023, learning rate 0.0282, Adam beta values 0.55 and 0.9942, and per-epoch validation/best-epoch selection. Do not add grouped windows, a handwritten/analytic gradient, SIMD, host-side Adam, or other optimizer speedups to the baseline.
- **Timed region:** only the Rust `compute_parameters()` call, timed with a monotonic clock inside the Python binding. Python input conversion and evaluation are excluded.
- **Accuracy:** after timing, evaluate the trained parameters on the user's same items with the frozen Rust evaluator. The arithmetic mean of the 50 per-user LogLoss values must be **[0.3078, 0.3113] inclusive**.
Keep the model and inputs fixed; otherwise the results are not comparable.
## Timing runs
Run each user **three times per build** and use the minimum Rust-call time for that user. Interference only adds time; the minimum is a less biased estimate of the compute floor. Report the median of per-user times. The supplied `--paired BASELINE CANDIDATE` harness reports the **median of paired per-user ratios** `baseline_ms / candidate_ms`, not the ratio of the two medians. Supply immutable, distinct extension files. Use a separate Cargo target directory for candidate builds so compiling a candidate never overwrites the baseline shared library. Store one output file per model using `--run-name`.
When the same machine can run both builds concurrently, use paired A/B timing:
1. Run baseline and candidate on the same user simultaneously on neighboring, disjoint two-logical-CPU blocks.
2. Run three repetitions; swap CPU blocks and which build starts first between repetitions.
3. Do not advance to the next user until both builds finish the current user.
4. Use each build's minimum of three timings; calculate per-user `baseline / candidate` ratios and their median.
5. Evaluate both builds after timing; confirm the candidate remains in-band and record the baseline/candidate accuracy difference.
This cancels much time-varying background load and avoids giving one build a quieter core or first-start advantage. If paired processes are unavailable, run baseline and candidate in alternating order in one session, three times each, and label the timings unpaired. Do not treat runs from different machines or substantially different load conditions as paired.
### Low-noise setup (Windows)
The supplied harness raises worker priority and pins workers to disjoint CPU pairs when supported. To reduce clock drift, Windows users can lock the AC processor state to 99% (administrator rights may be needed):
```powershell
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 99
powercfg /setactive SCHEME_CURRENT
```
Close unrelated CPU-intensive programs. Record CPU, OS, Python/Rust versions, date, process count, and whether timing was paired.
## Leaderboard fields
For every model/experiment (at most 20), preserve the source revision and record: experiment number, model/provider and slug, prompt/version, API cost in USD and available token counts, median paired speedup, per-user timing summary, mean LogLoss, baseline LogLoss, and machine/software environment. A change outside the accuracy band is **not a valid speedup**. Report API cost separately from optimizer speed.
## Interpretation
This is the expensive optimizer that preceded the original accepted speedups, paired with the finished 34-parameter FSRS-7 recurrence. It is a common starting line for independent experiments, not a claim about current FSRS-rs performance. Keep timing overhead and dataset preprocessing out of the Rust-call timing. Small differences near 1.0 require repeated paired sessions; minimum-of-three and paired medians reduce noise but cannot remove all uncertainty.

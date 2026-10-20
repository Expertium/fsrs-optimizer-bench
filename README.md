# FSRS Optimizer Bench

A self-contained, reproducible benchmark for experiments that make the FSRS parameter optimizer faster. It uses the slow, pre-accepted-speedup Burn/autodiff optimizer with the later, finished 34-parameter FSRS-7 model.

## What is being benchmarked
- Training uses the original Burn/autodiff optimizer from before the first accepted speedup (history iteration 2). It retains the autodiff training and validation passes and per-epoch best-model selection.
- It uses the finished FSRS-7 training operating point: 9 epochs, batch size 256, seed 2023, learning rate 0.0282, Adam beta values 0.55 and 0.9942. These settings are held fixed across experiments.
- The model has separate long- and short-term stability traces, short-trace recall in both the mixture and its stability update, surprise-weighted difficulty updates on lapses, and the post-lapse short-trace reset. The equations are checked against [`models/fsrs_v7.py`](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/models/fsrs_v7.py).
- Accuracy gate: mean LogLoss on users 1-50 must be in **[0.3066, 0.3096]** (0.3081 ± 0.0015). The canonical run checks this range.
This repository intentionally does **not** include the original project's complexity scorer or speedups. It benchmarks how much a model accelerates this fixed optimizer.
## Requirements and setup

Use Windows 10/11 x64, Python 3.12+, `uv`, and Rust 1.85 (pinned in `fsrs-rs/rust-toolchain.toml`). The repository includes the revlog partitions for users 1–50, so no Hugging Face download is needed. This is the corresponding subset of the public [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k) dataset; only numeric review logs are included. To use a different copy, pass its dataset root with `--data`.

From the repository directory:

```powershell
uv sync
uv run python -m pytest -q
cargo +1.85 fmt --manifest-path fsrs-rs/Cargo.toml --all -- --check
```

`uv sync` installs the project and test dependencies from the lockfile. The local Rust extension is built automatically on first import using the pinned Rust toolchain and the CPython stable ABI (Python 3.12+). The bundled data includes only numeric review-log fields; card IDs are factorized within each user as in the source dataset.

## Run the canonical 50-user benchmark

```powershell
uv run python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50 --run-name baseline
```

Each user is optimized three times; the minimum of those Rust-call timings is recorded, and LogLoss is evaluated afterwards. The canonical run checks exactly 50 users, **1,897,936 items**, and mean LogLoss in **[0.3066, 0.3096]** (0.3081 ± 0.0015). Results go to `result/compute_parameters-FSRS-rs-short-secs-recency-baseline.jsonl`; the summary is printed at completion. A distinct `--run-name` keeps each run separate. If a run is interrupted, rerun the same command to resume missing users; delete that named result file first if you intentionally want a clean repeat. Paired comparisons save per-user progress and resume only when the build hashes, dataset path, user set, and repeat count still match.

The default dataset location is the repository's `data/` directory. Override it with `--data "D:\\datasets\\anki-revlogs-10k"` if needed.

## Compare a candidate with the baseline

Build both versions in separate Cargo target directories so one build cannot overwrite the other. Preserve the known-good baseline extension before rebuilding or changing the optimizer:

```powershell
New-Item -ItemType Directory -Force profiling/measure | Out-Null
Copy-Item fsrs_rs_python/target/release/fsrs_rs_python.abi3.pyd profiling/measure/baseline.abi3.pyd
$env:CARGO_TARGET_DIR = "fsrs_rs_python/target/experiment-01"
cargo +1.85 build --manifest-path fsrs_rs_python/Cargo.toml --release --features pyo3/extension-module,pyo3/abi3-py312
Copy-Item fsrs_rs_python/target/experiment-01/release/fsrs_rs_python.dll profiling/measure/candidate.abi3.pyd
uv run python compute_parameters.py --algo FSRS-rs --short --secs --recency --max-user-id 50 --run-name experiment-01 --paired profiling/measure/baseline.abi3.pyd profiling/measure/candidate.abi3.pyd
```

The paired harness runs both builds on the same user's data concurrently, alternates their CPU blocks and start order across three repetitions, then reports the median per-user `baseline_ms / candidate_ms` ratio, both mean LogLoss values, and whether the candidate meets the accuracy band. Per-user timings and summary are saved as JSON under `result/`. Paired timing requires Windows and at least eight logical CPUs; use single-build runs in alternating order and clearly label them unpaired on other systems. See [DETAILS.md](DETAILS.md) for the protocol and leaderboard fields.

Python builds expanding-prefix items per card. It does not pass card IDs: card-grouped windows were added as an optimizer speedup and are absent from the baseline.

## Correctness and experiment rules

`tests/test_fsrs7_math.py` compares Rust memory-state trajectories against an independent scalar transcription of finished FSRS-7, including lapse surprise and the short-trace reset. Keep candidate changes focused on the Rust optimizer and preserve model equations, training settings, and benchmark inputs. For each of at most 20 experiments, record model/provider and slug, prompt/version, API cost and available token counts, source revision, paired speedup, per-user timing summary, LogLoss, and machine/software environment. A result outside the accuracy band is not a valid speedup.

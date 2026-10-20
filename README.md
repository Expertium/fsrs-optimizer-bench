# FSRS-Optimizer-Bench
A reproducible benchmark for experiments that make the FSRS parameter optimizer faster. It uses the slow, pre-accepted-speedup Burn/autodiff optimizer with the later, finished 34-parameter FSRS-7 model.
## What is being benchmarked
- Training uses the original Burn/autodiff optimizer from before the first accepted speedup (history iteration 2). It retains the autodiff training and validation passes and per-epoch best-model selection.
- It uses the finished FSRS-7 training operating point: 9 epochs, batch size 256, seed 2023, learning rate 0.0282, Adam beta values 0.55 and 0.9942. These settings are held fixed across experiments.
- The model has separate long- and short-term stability traces, short-trace recall in both the mixture and its stability update, surprise-weighted difficulty updates on lapses, and the post-lapse short-trace reset. The equations are checked against [`models/fsrs_v7.py`](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/models/fsrs_v7.py).
- Accuracy gate: mean LogLoss on users 1-50 must be in **[0.3078, 0.3113]**. The canonical run checks this range.
This repository intentionally does **not** include the original project's complexity scorer or speedups. It benchmarks how much a model accelerates this fixed optimizer.
## Run the canonical 50-user benchmark
Requirements: stable Rust 1.97+, Python 3.12+, and the `anki-revlogs-10k` dataset beside this repository (`../anki-revlogs-10k`). The dataset is not included.
```powershell
uv sync
uv run python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50 --run-name baseline
```
The local Python extension uses the CPython stable ABI (3.12+). Each user is optimized three times; the minimum of those three Rust-call timings is recorded, and LogLoss is evaluated afterwards. Results are written to `result/compute_parameters-FSRS-rs-short-secs-recency-baseline.jsonl`. `--run-name` keeps separate experiment results from replacing the baseline.
Build a candidate extension into a separate target directory and compare paired builds with the same 50 users:
```powershell
$env:CARGO_TARGET_DIR = "target\experiment-01"
cargo build --manifest-path fsrs_rs_python\Cargo.toml --release --features pyo3/extension-module,pyo3/abi3-py312
Copy-Item target\experiment-01\release\fsrs_rs_python.dll profiling\measure\candidate.abi3.pyd
uv run python compute_parameters.py --algo FSRS-rs --short --secs --recency --max-user-id 50 --run-name experiment-01 --paired target\release\fsrs_rs_python.abi3.pyd profiling\measure\candidate.abi3.pyd
```
The paired harness alternates the CPU blocks and which build starts first across repetitions; it reports the median of per-user `baseline_ms / candidate_ms` ratios, both mean LogLoss values, and whether the candidate meets the band. The baseline binary can be explicitly named as above if you have rebuilt the main target. See [DETAILS.md](DETAILS.md).
Python builds expanding-prefix items per card. It does not pass card IDs: card-grouped windows were added as an optimizer speedup and are absent from the baseline.
## Correctness test
```powershell
uv run --with pytest python -m pytest -q
```
`tests/test_fsrs7_math.py` compares the Rust memory-state trajectory against an independent scalar transcription of finished FSRS-7, including lapse surprise and the short-trace reset.
## Measurement and experiment rules
See [`DETAILS.md`](DETAILS.md) for the paired speed-measurement protocol, accuracy band, noise controls, and leaderboard fields. Keep candidate changes focused on the Rust optimizer and preserve model equations and benchmark inputs. Record model identity, experiment number, API spend/tokens, measured speedup, and LogLoss.

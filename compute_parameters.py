"""Time Rust compute_parameters() per user and score it with Rust evaluate().

Counterpart to benchmark.py, but:
  * trains with compute_parameters() (the function the speed campaign optimizes),
  * scores with Rust evaluate() instead of the Python forgetting curve,
  * uses train set == test set (no time-series split), so log loss is expected
    to be *lower* than benchmark.py's cross-validated number,
  * times each user's compute_parameters() 3x and records the MIN (ms). Timing
    noise is one-sided (interference only ever slows a run), so the fastest of
    the three is the cleanest estimate of the true compute floor.

The timed region is the Rust compute_parameters() call only (monotonic clock,
measured inside the extension); the Python here stays untimed.
"""

import json
import os
import platform
import statistics
import sys
import hashlib
import shutil
import tempfile
from pathlib import Path as _Path

# Ensure the repo root is first on sys.path so local packages resolve even when
# this is invoked from another directory.
_REPO_ROOT = str(_Path(__file__).resolve().parent)
if not sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from config import create_parser, Config
from data_loader import UserDataLoader
from utils import catch_exceptions, sort_jsonl
_FSRS_MODULE = None
def _fsrs_module():
    global _FSRS_MODULE
    if _FSRS_MODULE is None:
        import fsrs_rs_python
        _FSRS_MODULE = fsrs_rs_python
    return _FSRS_MODULE

# Number of times compute_parameters() is timed per user; the MIN is recorded.
TIMING_REPEATS = 3
# ---------------------------------------------------------------------------
# Review-history parsing (same shape as benchmark.py)
# ---------------------------------------------------------------------------


def _parse_scalar(value: object, type_name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Expected a {type_name} history value, got missing data")
    result = str(value).strip()
    if not result:
        raise ValueError(f"Expected a {type_name} history value, got empty text")
    return result


def parse_interval(value: object) -> float:
    """Parse a single delta_t field, clamped to be non-negative."""
    return max(0.0, float(_parse_scalar(value, "numeric review")))


def parse_rating(value: object) -> int:
    return int(float(_parse_scalar(value, "rating")))


def parse_history(history: object, parser) -> list:
    """Split a comma-separated history string and parse each non-empty field."""
    if pd.isna(history):
        return []
    fields = (field.strip() for field in str(history).split(","))
    return [parser(field) for field in fields if field]


def build_reviews(row: pd.Series, *, include_current: bool = False) -> list:
    t_history = parse_history(row["t_history"], parse_interval)
    r_history = parse_history(row["r_history"], parse_rating)
    if include_current:
        t_history.append(parse_interval(row["delta_t"]))
        r_history.append(parse_rating(row["rating"]))
    if len(t_history) != len(r_history):
        raise ValueError("Review history lengths do not match")
    review_type = _fsrs_module().FSRSReview
    return [review_type(delta_t=t, rating=r) for t, r in zip(t_history, r_history)]


def convert_to_items(df: pd.DataFrame) -> tuple[list, List[int]]:
    """Convert a DataFrame to FSRSItems for fsrs-rs, ordered globally by review_th.

    Also returns each item's originating card id (parallel list, same order) so the optimizer can
    group a card's expanding-window prefix-items into one mini-batch (the O(N) window path). The
    items list and ordering are unchanged vs. before, so log loss / timing are unaffected by this.
    """
    item_type = _fsrs_module().FSRSItem
    pairs = []  # (review_th, card_id, FSRSItem)
    for card_id, group in df.sort_values(by=["card_id", "review_th"]).groupby("card_id"):
        for _, row in group.iterrows():
            item = item_type(reviews=build_reviews(row, include_current=True))
            pairs.append((row["review_th"], int(card_id), item))
    # Sort by review_th only (the key); ties keep insertion order and never compare the items.
    pairs.sort(key=lambda pair: pair[0])
    items = [item for _, _, item in pairs]
    card_ids = [cid for _, cid, _ in pairs]
    return items, card_ids


parser = create_parser()
parser.add_argument("--run-name", default=None, help="separate results for this experiment")
parser.add_argument("--paired", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                    help="compare two Python extension builds on identical users")
args, _ = parser.parse_known_args()
if args.gpus is not None:
    raise ValueError("GPU execution is prohibited by the benchmark protocol")
if args.algo == parser.get_default("algo"):
    args.algo = "FSRS-rs"
elif args.algo != "FSRS-rs":
    raise ValueError("compute_parameters only supports --algo FSRS-rs")
config = Config(args)
config.partitions = "none"

# Distinct output name so results never collide with benchmark.py's.
OUTPUT_NAME = f"compute_parameters-{config.get_evaluation_file_name()}"
if args.paired:
    OUTPUT_NAME += "-paired"
if args.run_name:
    OUTPUT_NAME += f"-{args.run_name}"


@catch_exceptions
def process(user_id: int) -> tuple[dict, Optional[dict]]:
    """Train one user's parameters and score that user's reviews."""
    dataset = UserDataLoader(config).load_user_data(user_id)
    items, _card_ids = convert_to_items(dataset)

    backend = _fsrs_module().FSRS(parameters=[])
    times_ms = []
    params: List[float] = []
    for _ in range(TIMING_REPEATS):
        params, secs = backend.compute_parameters(items)
        times_ms.append(secs * 1000.0)
    params = [round(w, 4) for w in params]

    predictor = _fsrs_module().FSRS(parameters=params)
    log_loss = predictor.evaluate(items)  # train set == test set

    stats = {
        "user": int(user_id),
        "time_ms": round(min(times_ms)),
        "time_ms_runs": [round(t) for t in times_ms],
        "parameters": params,
        "metrics": {"LogLoss": round(log_loss, 6)},
        "size": len(items),
    }
    return stats, None


def _paired_worker(extension_path, connection):
    os.environ["FSRS_EXTENSION_PATH"] = str(extension_path)
    import fsrs_rs_python as fsrs_module
    connection.send("ready")
    cached_user = None
    cached_items = None
    while True:
        command = connection.recv()
        if command is None:
            break
        kind, user_id, *extra = command
        try:
            if user_id != cached_user:
                cached_items = convert_to_items(
                    UserDataLoader(config).load_user_data(user_id)
                )[0]
                cached_user = user_id
            if kind == "prepare":
                connection.send(("prepared", user_id, len(cached_items)))
            elif kind == "timed":
                params, elapsed = fsrs_module.FSRS(parameters=[]).compute_parameters(cached_items)
                connection.send((elapsed * 1000.0, params))
            elif kind == "score":
                scorer = fsrs_module.FSRS(parameters=extra[0])
                connection.send(scorer.evaluate(cached_items))
        except Exception as exc:
            connection.send(("error", repr(exc)))
def _write_json_atomically(path: Path, data: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary_path.replace(path)
def _write_paired_checkpoint(path: Path, metadata: dict, results: dict) -> None:
    _write_json_atomically(path, {"metadata": metadata, "per_user": results})
def _paired_summary(results: dict) -> dict:
    rows = list(results.values())
    baseline_loss = statistics.mean(row["baseline_logloss"] for row in rows)
    candidate_loss = statistics.mean(row["candidate_logloss"] for row in rows)
    return {
        "users": len(rows),
        "total_items": sum(row["baseline_size"] for row in rows),
        "candidate_total_items": sum(row["candidate_size"] for row in rows),
        "median_speedup": statistics.median(row["baseline_ms"] / row["candidate_ms"] for row in rows),
        "mean_speedup": statistics.mean(row["baseline_ms"] / row["candidate_ms"] for row in rows),
        "median_ms_baseline": statistics.median(row["baseline_ms"] for row in rows),
        "median_ms_candidate": statistics.median(row["candidate_ms"] for row in rows),
        "mean_logloss_baseline": baseline_loss,
        "mean_logloss_candidate": candidate_loss,
        "baseline_logloss_in_band": 0.3066 <= baseline_loss <= 0.3096 if len(rows) == 50 else None,
        "candidate_logloss_in_band": 0.3066 <= candidate_loss <= 0.3096 if len(rows) == 50 else None,
        "review_counts_match": all(row["baseline_size"] == row["candidate_size"] for row in rows),
        "review_count_mismatches": [
            user_id
            for user_id, row in results.items()
            if row["baseline_size"] != row["candidate_size"]
        ],
        "full_run": len(rows) == 50,
    }
def _write_paired_result(results: dict, metadata: dict) -> dict:
    if not results:
        raise RuntimeError("no users completed both optimization and evaluation")
    summary = _paired_summary(results)
    if summary["full_run"] and summary["total_items"] != 1_897_936:
        raise AssertionError(f"expected 1,897,936 paired items, got {summary['total_items']:,}")
    reasons = []
    if summary["full_run"]:
        if not summary["review_counts_match"]:
            reasons.append("per-user processed review counts differ from baseline")
        if not summary["baseline_logloss_in_band"]:
            raise RuntimeError("pinned baseline failed the accuracy band; this run is invalid, not rejected")
        if not summary["candidate_logloss_in_band"]:
            reasons.append("candidate mean LogLoss outside [0.3066, 0.3096]")
        if summary["median_speedup"] < 1.015:
            reasons.append("median paired speedup below 1.5%")
    summary["rejection_reasons"] = reasons
    summary["status"] = ("accepted" if not reasons else "rejected") if summary["full_run"] else "incomplete"
    result_file = Path("result") / f"{OUTPUT_NAME}.json"
    _write_json_atomically(
        result_file,
        {
            "summary": summary,
            "per_user": results,
            "checkpoint_metadata": metadata,
            "baseline_sha256": metadata["baseline_sha256"],
            "candidate_sha256": metadata["candidate_sha256"],
        },
    )
    print(json.dumps(summary, indent=2))
    return summary
def _run_paired_benchmark(baseline_path: str, candidate_path: str) -> None:
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("paired measurements require x86-64 with baseline SSE2 support")
    if sys.platform != "win32":
        raise RuntimeError("paired measurements currently require Windows; use unpaired runs elsewhere")
    baseline_path = str(Path(baseline_path).resolve())
    candidate_path = str(Path(candidate_path).resolve())
    if Path(baseline_path) == Path(candidate_path):
        raise ValueError("baseline and candidate must be different extension files")
    for extension_path in (baseline_path, candidate_path):
        if not Path(extension_path).is_file():
            raise FileNotFoundError(f"extension file not found: {extension_path}")
    data = pq.ParquetDataset(config.data_path / "revlogs")
    users = sorted(
        entry.as_py()
        for entry in data.partitioning.dictionaries[0]
        if config.max_user_id is None or entry.as_py() <= config.max_user_id
    )
    if config.max_user_id == 50 and users != list(range(1, 51)):
        raise ValueError("expected exactly users 1-50")
    checkpoint_file = Path("result") / f"{OUTPUT_NAME}-partial.json"
    checkpoint_metadata = {
        "baseline_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
        "data_path": str(config.data_path.resolve()),
        "users": users,
        "timing_repeats": TIMING_REPEATS,
    }
    results = {}
    if checkpoint_file.is_file():
        try:
            saved = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if saved.get("metadata") == checkpoint_metadata:
                results = {
                    int(user_id): row
                    for user_id, row in saved.get("per_user", {}).items()
                    if int(user_id) in users
                    and len(row.get("baseline_runs_ms", [])) == TIMING_REPEATS
                    and len(row.get("candidate_runs_ms", [])) == TIMING_REPEATS
                    and isinstance(row.get("baseline_logloss"), (int, float))
                    and isinstance(row.get("candidate_logloss"), (int, float))
                    and isinstance(row.get("baseline_size"), int)
                    and isinstance(row.get("candidate_size"), int)
                }
        except (OSError, ValueError, TypeError):
            results = {}
    remaining_users = [user_id for user_id in users if user_id not in results]
    if not remaining_users:
        print(f"Resuming completed paired data from {checkpoint_file}", flush=True)
        _write_paired_result(results, checkpoint_metadata)
        return
    with tempfile.TemporaryDirectory(prefix="fsrs_paired_") as temp_dir:
        workers = []
        connections = []
        try:
            for label, extension in (("baseline", baseline_path), ("candidate", candidate_path)):
                staged_dir = Path(temp_dir) / label
                staged_dir.mkdir()
                staged = staged_dir / Path(extension).name
                shutil.copy2(extension, staged)
                parent, child = Pipe()
                worker = Process(target=_paired_worker, args=(str(staged), child))
                worker.start()
                workers.append(worker)
                connections.append(parent)
            for connection in connections:
                if not connection.poll(1800):
                    raise RuntimeError("paired worker startup failed; fix and retry this experiment")
                if connection.recv() != "ready":
                    raise RuntimeError("paired worker initialization failed")
            for user_index, user_id in enumerate(remaining_users):
                for connection in connections:
                    connection.send(("prepare", user_id))
                prepared = []
                for connection in connections:
                    if not connection.poll(1800):
                        raise RuntimeError(f"input preparation failed for user {user_id}; retry this experiment")
                    prepared.append(connection.recv())
                for reply in prepared:
                    if not isinstance(reply, tuple) or reply[0] != "prepared" or reply[1] != user_id:
                        detail = reply[1] if isinstance(reply, tuple) and len(reply) > 1 else reply
                        raise RuntimeError(f"input preparation failed for user {user_id}: {detail}")
                timings = [[], []]
                parameters = [None, None]
                for repeat in range(TIMING_REPEATS):
                    launch_order = [0, 1] if (repeat + user_index) % 2 == 0 else [1, 0]
                    for worker_index in launch_order:
                        connections[worker_index].send(("timed", user_id))
                    for connection in connections:
                        if not connection.poll(1800):
                            raise RuntimeError(f"optimizer failed or timed out for user {user_id}; retry this experiment")
                    for worker_index, connection in enumerate(connections):
                        reply = connection.recv()
                        if isinstance(reply, tuple) and reply and reply[0] == "error":
                            raise RuntimeError(f"optimizer failed for user {user_id}: {reply[1]}")
                        elapsed_ms, parameters[worker_index] = reply
                        if elapsed_ms <= 0:
                            raise RuntimeError(f"optimizer returned a non-positive timing for user {user_id}")
                        timings[worker_index].append(elapsed_ms)
                for worker_index, connection in enumerate(connections):
                    connection.send(("score", user_id, parameters[worker_index]))
                losses = []
                for connection in connections:
                    if not connection.poll(1800):
                        raise RuntimeError(f"evaluation failed or timed out for user {user_id}; retry this experiment")
                    losses.append(connection.recv())
                for reply in losses:
                    if isinstance(reply, tuple) and reply and reply[0] == "error":
                        raise RuntimeError(f"evaluation failed for user {user_id}: {reply[1]}")
                results[user_id] = {
                    "size": prepared[0][2],
                    "baseline_size": prepared[0][2],
                    "candidate_size": prepared[1][2],
                    "baseline_runs_ms": timings[0],
                    "candidate_runs_ms": timings[1],
                    "baseline_ms": min(timings[0]),
                    "candidate_ms": min(timings[1]),
                    "baseline_logloss": losses[0],
                    "candidate_logloss": losses[1],
                }
                _write_paired_checkpoint(checkpoint_file, checkpoint_metadata, results)
                print(f"paired user {user_id}/{len(users)}", flush=True)
        finally:
            for connection in connections:
                try:
                    connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
    _write_paired_result(results, checkpoint_metadata)
if __name__ == "__main__":
    if args.paired:
        _run_paired_benchmark(*args.paired)
        raise SystemExit(0)
    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    Path("result").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{OUTPUT_NAME}.jsonl")
    if result_file.exists():
        processed_user = {row["user"] for row in sort_jsonl(result_file)}
    else:
        processed_user = set()

    unprocessed_users = []
    for user_id in dataset.partitioning.dictionaries[0]:
        user_id_value = user_id.as_py()
        if config.max_user_id is not None and user_id_value > config.max_user_id:
            continue
        if user_id_value in processed_user:
            continue
        unprocessed_users.append(user_id_value)
    unprocessed_users.sort()

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(process, user_id) for user_id in unprocessed_users]
        for future in (
            pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
                    stats, _ = result
                    with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    data = sort_jsonl(result_file)
    n_users = len(data)
    if n_users:
        total_items = sum(d["size"] for d in data)
        mean_ll = sum(d["metrics"]["LogLoss"] for d in data) / n_users
        # Median per-user time is the metric of record; mean log
        # loss is the conventional correctness aggregate that matches the references.
        median_time = statistics.median(d["time_ms"] for d in data)
        print(
            f"users={n_users} total_items={total_items} "
            f"LogLoss(mean)={mean_ll:.4f} time(median)={median_time:.1f}ms"
        )
    # Number of users AND dataset size must stay fixed for the canonical run
    # (constraint 5), so review preprocessing can't silently change.
    if config.max_user_id == 50:
        assert n_users == 50, f"expected 50 users, got {n_users}"
        assert total_items == 1_897_936, f"expected 1,897,936 items, got {total_items:,}"
        assert 0.3066 <= mean_ll <= 0.3096, (
            f"mean LogLoss {mean_ll:.6f} is outside [0.3066, 0.3096]"
        )

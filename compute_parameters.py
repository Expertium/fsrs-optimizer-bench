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

To reduce measurement noise, launch with high priority pinned off CPUs 0-3:

    start /high /affinity 0xFFFFFFF0 python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50

On top of that, each pool worker pins itself to its own disjoint 2-CPU block and
re-applies HIGH priority (workers inherit the launch's affinity but not its
priority class), so cross-worker contention is constant rather than random.
"""

import json
import os
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
from multiprocessing import Manager, Pipe, Process, Queue
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
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
# Worker CPU pinning: leave logical CPUs 0-3 for the OS, give each worker its own
# disjoint block of CPUs (<= 2, per the "<=2 threads per user" constraint).
_PIN_RESERVED_CPUS = 4
_PIN_CPUS_PER_WORKER = 2

# Canonical 50-user processing order, biggest collection first -> smallest last
# (sizes = item counts from result/compute_parameters-FSRS-rs-short-secs-recency.jsonl,
# the --short --secs --recency config this campaign runs). Dispatching the largest
# users first makes them double as a CPU warm-up: by the time the small,
# noise-sensitive users (which dominate the median metric, constraint 12) run, the
# machine is at thermal/cache equilibrium instead of cold. This is a pure
# measurement-noise heuristic -- it changes only scheduling order, not the recorded
# params/log loss (sort_jsonl re-sorts the output by user at the end anyway).
USERS_BY_SIZE_DESC = [
    34, 17, 33, 40, 15, 6, 41, 14, 12, 21,
    25, 2, 38, 24, 8, 39, 32, 35, 18, 10,
    47, 20, 43, 28, 31, 37, 1, 30, 44, 11,
    7, 45, 5, 50, 29, 13, 4, 48, 23, 16,
    19, 27, 49, 3, 46, 9, 22, 36, 26, 42,
]
assert len(USERS_BY_SIZE_DESC) == len(set(USERS_BY_SIZE_DESC)), \
    "duplicate user id in USERS_BY_SIZE_DESC"


def _set_high_priority() -> None:
    """Raise this worker to HIGH priority. Workers inherit the launch's CPU
    affinity but NOT its priority class, so we re-apply it here. HIGH needs no
    admin rights (unlike REALTIME). No-op off Windows; never raises."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Type the handle explicitly: an untyped GetCurrentProcess return is
        # marshalled as a 32-bit int and SetPriorityClass then silently fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass


def _pin_to_cpus(worker_index: int) -> None:
    """Pin a worker to its own disjoint CPU pair when the host supports it."""
    _set_cpu_affinity(
        range(
            _PIN_RESERVED_CPUS + worker_index * _PIN_CPUS_PER_WORKER,
            _PIN_RESERVED_CPUS + (worker_index + 1) * _PIN_CPUS_PER_WORKER,
        )
    )
def _set_cpu_affinity(cpus) -> None:
    if sys.platform != "win32":
        return
    cpus = tuple(cpus)
    if not cpus or max(cpus) >= (os.cpu_count() or 0):
        return
    mask = sum(1 << cpu for cpu in cpus)
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask)
    except Exception:
        pass


def _init_worker(counter, lock) -> None:
    """Pool worker initializer: HIGH priority + a unique disjoint CPU block."""
    _set_high_priority()
    with lock:
        worker_index = counter.value
        counter.value = worker_index + 1
    _pin_to_cpus(worker_index)


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
if args.algo == parser.get_default("algo"):
    args.algo = "FSRS-rs"
elif args.algo != "FSRS-rs":
    raise ValueError("compute_parameters only supports --algo FSRS-rs")
config = Config(args)
config.partitions = "none"

torch.manual_seed(config.seed)

# Distinct output name so results never collide with benchmark.py's.
OUTPUT_NAME = f"compute_parameters-{config.get_evaluation_file_name()}"
if args.run_name:
    OUTPUT_NAME += f"-{args.run_name}"
if args.paired:
    OUTPUT_NAME += "-paired-" + (args.run_name or "run")


@catch_exceptions
def process(
    user_id: int,
    device_id: Optional[int] = None,
    timing_cpus: Optional[List[List[int]]] = None,
) -> tuple[dict, Optional[dict]]:
    """Train with compute_parameters() and score with evaluate(); train set == test set."""
    del device_id
    dataset = UserDataLoader(config).load_user_data(user_id)
    items, _card_ids = convert_to_items(dataset)

    backend = _fsrs_module().FSRS(parameters=[])
    # compute_parameters() is deterministic (seeded), so params are identical
    # across repeats; only the timing varies. Record the min (cleanest) time.
    times_ms = []
    params: List[float] = []
    for repeat in range(TIMING_REPEATS):
        if timing_cpus:
            _set_cpu_affinity(timing_cpus[repeat])
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
    while True:
        command = connection.recv()
        if command is None:
            break
        kind, user_id, cpus, *extra = command
        _set_high_priority()
        if cpus is not None:
            _set_cpu_affinity(cpus)
        try:
            items, _ = convert_to_items(UserDataLoader(config).load_user_data(user_id))
            if kind == "timed":
                params, elapsed = fsrs_module.FSRS(parameters=[]).compute_parameters(items)
                connection.send((elapsed * 1000.0, [round(w, 4) for w in params]))
            else:
                scorer = fsrs_module.FSRS(parameters=extra[0])
                connection.send(scorer.evaluate(items))
        except Exception as exc:
            connection.send(("error", repr(exc)))
def _run_paired_benchmark(baseline_path: str, candidate_path: str) -> None:
    baseline_path = str(Path(baseline_path).resolve())
    candidate_path = str(Path(candidate_path).resolve())
    if not Path(baseline_path).is_file() or not Path(candidate_path).is_file():
        raise FileNotFoundError("Both paired extension paths must name built extension files")
    data = pq.ParquetDataset(config.data_path / "revlogs")
    users = [entry.as_py() for entry in data.partitioning.dictionaries[0]
             if config.max_user_id is None or entry.as_py() <= config.max_user_id]
    _size_rank = {uid: i for i, uid in enumerate(USERS_BY_SIZE_DESC)}
    users.sort(key=lambda uid: (_size_rank.get(uid, len(_size_rank)), uid))
    if config.max_user_id == 50 and len(users) != 50:
        raise ValueError(f"expected 50 users, got {len(users)}")
    with tempfile.TemporaryDirectory(prefix="fsrs_paired_") as temp_dir:
        paths = []
        for label, source in (("baseline", baseline_path), ("candidate", candidate_path)):
            target = Path(temp_dir) / label
            target.mkdir()
            ext = Path(source)
            staged = target / ext.name
            shutil.copy2(ext, staged)
            paths.append(staged)
        workers = []
        connections = []
        for path in paths:
            parent, child = Pipe()
            worker = Process(target=_paired_worker, args=(str(path), child))
            worker.start()
            workers.append(worker)
            connections.append(parent)
        for conn in connections:
            if conn.recv() != "ready":
                raise RuntimeError("paired worker failed to initialize")
        results = {}
        first_cpu = 4
        for user_index, user_id in enumerate(users):
            timings = [[], []]
            params = [None, None]
            for repeat in range(TIMING_REPEATS):
                first_block = [first_cpu, first_cpu + 1]
                second_block = [first_cpu + 2, first_cpu + 3]
                swap = repeat % 2
                blocks = [first_block, second_block]
                if swap:
                    blocks.reverse()
                start_order = [0, 1] if (repeat + user_index) % 2 == 0 else [1, 0]
                for worker_index in start_order:
                    connections[worker_index].send(("timed", user_id, blocks[worker_index]))
                for worker_index in range(2):
                    reply = connections[worker_index].recv()
                    if isinstance(reply, tuple) and reply and reply[0] == "error":
                        raise RuntimeError(f"user {user_id}: {reply[1]}")
                    milliseconds, params[worker_index] = reply
                    timings[worker_index].append(milliseconds)
            results[user_id] = {
                "a_runs_ms": timings[0], "b_runs_ms": timings[1],
                "a_ms": min(timings[0]), "b_ms": min(timings[1]),
                "a_parameters": params[0], "b_parameters": params[1],
            }
            for worker_index, connection in enumerate(connections):
                connection.send(("score", user_id, None, params[worker_index]))
            for worker_index, connection in enumerate(connections):
                loss = connection.recv()
                if isinstance(loss, tuple) and loss and loss[0] == "error":
                    raise RuntimeError(f"user {user_id}: {loss[1]}")
                results[user_id]["a_logloss" if worker_index == 0 else "b_logloss"] = loss
            print(f"paired user {user_id}/{len(users)}", flush=True)
        for connection in connections:
            connection.send(None)
        for worker in workers:
            worker.join()
    rows = list(results.values())
    ratios = [row["a_ms"] / row["b_ms"] for row in rows]
    mean_a = statistics.mean(row["a_logloss"] for row in rows)
    mean_b = statistics.mean(row["b_logloss"] for row in rows)
    summary = {
        "users": len(rows),
        "median_speedup": statistics.median(ratios),
        "mean_speedup": statistics.mean(ratios),
        "median_ms_baseline": statistics.median(row["a_ms"] for row in rows),
        "median_ms_candidate": statistics.median(row["b_ms"] for row in rows),
        "mean_logloss_baseline": mean_a,
        "mean_logloss_candidate": mean_b,
        "candidate_logloss_in_band": (0.3078 <= mean_b <= 0.3113) if len(rows) == 50 else None,
        "parameter_hash_diffs": sum(
            hashlib.sha256(repr(row["a_parameters"]).encode()).digest()
            != hashlib.sha256(repr(row["b_parameters"]).encode()).digest()
            for row in rows
        ),
    }
    Path("result").mkdir(exist_ok=True)
    result_file = Path(f"result/{OUTPUT_NAME}.json")
    result_file.write_text(json.dumps({"summary": summary, "per_user": results}, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["candidate_logloss_in_band"] is False:
        raise AssertionError(
            f"candidate mean LogLoss {mean_b:.6f} outside [0.3078, 0.3113]"
        )
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
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
    # Dispatch biggest collections first so they warm up the CPU (see
    # USERS_BY_SIZE_DESC). ProcessPoolExecutor pulls submitted tasks FIFO, so
    # submission order == start order. Any user not in the hardcoded ranking
    # (e.g. --max-user-id > 50) sorts last, by ascending id, so behaviour stays
    # deterministic for non-canonical runs.
    _size_rank = {uid: i for i, uid in enumerate(USERS_BY_SIZE_DESC)}
    unprocessed_users.sort(key=lambda uid: (_size_rank.get(uid, len(_size_rank)), uid))

    # Hand each worker a unique index (via a manager-backed counter) so it can
    # claim its own disjoint CPU block in the initializer.
    manager = mp.Manager()
    worker_counter = manager.Value("i", 0)
    counter_lock = manager.Lock()

    with ProcessPoolExecutor(
        max_workers=config.num_processes,
        initializer=_init_worker,
        initargs=(worker_counter, counter_lock),
    ) as executor:
        futures = [executor.submit(process, user_id, None) for user_id in unprocessed_users]
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
        assert 0.3078 <= mean_ll <= 0.3113, (
            f"mean LogLoss {mean_ll:.6f} is outside [0.3078, 0.3113]"
        )

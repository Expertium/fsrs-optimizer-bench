import compute_parameters


def _row(user_id, candidate_size=None):
    baseline_size = 37_959 if user_id <= 36 else 37_958
    if candidate_size is None:
        candidate_size = baseline_size
    return {
        "size": baseline_size,
        "baseline_size": baseline_size,
        "candidate_size": candidate_size,
        "baseline_ms": 102.0,
        "candidate_ms": 100.0,
        "baseline_logloss": 0.308,
        "candidate_logloss": 0.308,
    }


def test_review_count_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compute_parameters, "OUTPUT_NAME", "count-mismatch")
    results = {1: _row(1, candidate_size=37_958)}
    metadata = {"baseline_sha256": "base", "candidate_sha256": "candidate"}

    summary = compute_parameters._write_paired_result(results, metadata)

    assert summary["review_counts_match"] is False
    assert summary["review_count_mismatches"] == [1]
    assert summary["status"] == "incomplete"
    assert summary["rejection_reasons"] == []


def test_full_run_count_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compute_parameters, "OUTPUT_NAME", "count-mismatch")
    results = {user: _row(user) for user in range(1, 51)}
    results[17] = _row(17, candidate_size=37_958)
    results[18] = _row(18, candidate_size=37_960)
    metadata = {"baseline_sha256": "base", "candidate_sha256": "candidate"}

    summary = compute_parameters._write_paired_result(results, metadata)

    assert summary["users"] == 50
    assert summary["review_counts_match"] is False
    assert summary["review_count_mismatches"] == [17, 18]
    assert summary["status"] == "rejected"
    assert "per-user processed review counts differ from baseline" in summary[
        "rejection_reasons"
    ]


def test_incomplete_count_mismatch_is_not_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compute_parameters, "OUTPUT_NAME", "count-mismatch")
    results = {1: _row(1, candidate_size=37_958)}
    metadata = {"baseline_sha256": "base", "candidate_sha256": "candidate"}

    summary = compute_parameters._write_paired_result(results, metadata)

    assert summary["review_counts_match"] is False
    assert summary["status"] == "incomplete"
    assert summary["rejection_reasons"] == []

import math
from fsrs_rs_python import DEFAULT_PARAMETERS, FSRS, FSRSItem, FSRSReview
def _clamp(value, low, high):
    return max(low, min(value, high))
def _fast_recall(t, short_stability, w):
    t = max(0.0, t)
    decay = -_clamp(w[23] * short_stability ** (w[33] - 0.3), 0.01, 0.95)
    factor = math.exp(min(math.log(w[25]) / decay, 60.0)) - 1.0
    return (t / short_stability * factor + 1.0) ** decay
def _forgetting_curve(t, stability, short_stability, difficulty, w):
    t = max(t, 0.0)
    r1 = _fast_recall(t, short_stability, w)
    decay2 = -_clamp(w[24], 0.01, 0.95)
    factor2 = w[26] ** (1.0 / decay2) - 1.0
    d_timescale = math.exp((difficulty - 5.0) * (w[32] - 0.3))
    r2 = (t / stability * factor2 * d_timescale + 1.0) ** decay2
    weight1 = w[27] * short_stability ** (-w[29])
    weight2 = w[28] * stability**w[30] * math.exp(
        (difficulty - 5.0) * (w[31] - 0.5)
    )
    return 1e-5 + (1.0 - 2e-5) * (
        (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
    )
def _next_stability(stability, difficulty, recall, rating, start, w):
    hard = w[start + 6] if rating == 2 else 1.0
    easy = w[start + 7] if rating == 4 else 1.0
    after_lapse = w[start + 3] * (
        (stability + 1.0) ** w[start + 4] - 1.0
    ) * math.exp((1.0 - recall) * w[start + 5])
    post_lapse = min(stability, after_lapse)
    growth = (
        math.exp(w[start] - 1.5)
        * (11.0 - difficulty)
        * stability ** (-w[start + 1])
        * (math.exp((1.0 - recall) * w[start + 2]) - 1.0)
        * hard
        * easy
        + 1.0
    )
    if rating == 1:
        return post_lapse
    return max(post_lapse, stability * growth)
def _reference_state(reviews, w):
    stability = short_stability = difficulty = 0.0
    for index, (rating, delta_t) in enumerate(reviews):
        if index == 0:
            stability = w[rating - 1]
            short_stability = 0.8 * stability
            difficulty = _clamp(
                w[4] - math.exp(w[5] * (rating - 1)) + 1.0, 1.0, 10.0
            )
            continue
        stability = _clamp(stability, 0.0001, 36500.0)
        short_stability = _clamp(short_stability, 0.0001, 36500.0)
        difficulty = _clamp(difficulty, 1.0, 10.0)
        recall = _forgetting_curve(
            delta_t, stability, short_stability, difficulty, w
        )
        next_stability = _next_stability(
            stability, difficulty, recall, rating, 7, w
        )
        short_recall = _fast_recall(delta_t, short_stability, w)
        next_short_stability = _next_stability(
            short_stability, difficulty, short_recall, rating, 15, w
        )
        if rating == 1:
            next_short_stability = min(
                next_short_stability, 0.8 * next_stability
            )
        delta_d = -w[6] * (rating - 3)
        if rating == 1:
            delta_d *= recall + 0.1
        next_difficulty = difficulty + (10.0 - difficulty) * delta_d / 9.0
        init_d4 = w[4] - math.exp(w[5] * 3.0) + 1.0
        difficulty = _clamp(
            0.01 * init_d4 + 0.99 * next_difficulty, 1.0, 10.0
        )
        stability = _clamp(next_stability, 0.0001, 36500.0)
        short_stability = _clamp(next_short_stability, 0.0001, 36500.0)
    return stability, short_stability, difficulty
def test_finished_fsrs7_state_matches_reference_equations():
    histories = [
        [(3, 0.0), (2, 0.25), (1, 0.05), (4, 2.5), (3, 0.0)],
        [(1, 0.0), (1, 1.0), (2, 4.25), (4, 12.0)],
        [(4, 0.0), (3, 0.0005), (1, 3.0), (2, 0.75), (3, 8.0)],
    ]
    w = [float(value) for value in DEFAULT_PARAMETERS]
    items = [
        FSRSItem(reviews=[FSRSReview(rating, delta_t) for rating, delta_t in history])
        for history in histories
    ]
    states = FSRS(parameters=[]).memory_state_batch(items)
    for history, actual in zip(histories, states, strict=True):
        expected = _reference_state(history, w)
        assert math.isclose(actual.stability, expected[0], rel_tol=2e-5, abs_tol=2e-6)
        assert math.isclose(
            actual.stability_fast, expected[1], rel_tol=2e-5, abs_tol=2e-6
        )
        assert math.isclose(actual.difficulty, expected[2], rel_tol=2e-5, abs_tol=2e-6)

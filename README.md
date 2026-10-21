# FSRS Optimizer Bench

FSRS (Free Spaced Repetition Scheduler) schedules flashcard reviews by predicting how memory changes over time. Its optimizer fits a learner's review history to parameters that improve those predictions.

This benchmark compares AI-generated optimizer changes using a fixed, reproducible review-log dataset and accuracy requirements. Each accepted result must meet the benchmark's speed and LogLoss criteria; see [PROTOCOL.md](PROTOCOL.md) for setup, commands, and rules.

## Model leaderboard

| Model name | Reasoning effort | Harness | Total cumulative speedup | Acceptance rate | API costs |
|---|---|---|---:|---:|---:|

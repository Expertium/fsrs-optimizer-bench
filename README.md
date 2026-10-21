# FSRS Optimizer Bench

FSRS (Free Spaced Repetition Scheduler) schedules flashcard reviews by predicting how memory changes over time. Its optimizer fits a learner's review history to parameters that improve those predictions.

This benchmark compares AI-generated optimizer changes using a fixed, reproducible review-log dataset and accuracy requirements. Each accepted result must meet the benchmark's speed and LogLoss criteria; see [PROTOCOL.md](PROTOCOL.md) for setup, commands, and rules. 

If you would like to contribute to this benchmark, just give your agent the link to this repo and tell it to read PROTOCOL.md, then begin working. It is recommended to lock the frequency of your CPU to reduce measurement noise. On Windows 10, you can use the following commands:

```bat
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 99
powercfg /setactive SCHEME_CURRENT
```

## Model leaderboard

| Model name | Reasoning effort | Harness | Total cumulative speedup | Acceptance rate | API costs |
|---|---|---|---:|---:|---:|

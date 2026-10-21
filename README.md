# FSRS Optimizer Bench

FSRS (Free Spaced Repetition Scheduler) schedules flashcard reviews by predicting how memory changes over time. Its optimizer fits a learner's review history to parameters that improve those predictions. This benchmark compares AI-generated optimizer changes using a fixed, publicly available dataset of [Anki](https://github.com/ankitects/anki) users. See also https://github.com/Expertium/fsrs-rs-speed-autoresearch.

PR is welcome! Let your AI agent give it a shot and then make a PR. The details are in [AGENTS.md](/AGENTS.md) and [PROTOCOL.md](/PROTOCOL.md). No GPU is required, CPU only.

## Model leaderboard

| Model name | Reasoning effort | Harness | Total cumulative speedup | Acceptance rate | API costs |
|---|---|---|---:|---:|---:|
| --- | --- | --- | --- | --- | --- |

Total cumulative speedup is the overall accumulated improvement after 20 experiments. Acceptance rate is accepted / (accepted + rejected); error/incomplete attempts are excluded from both counts.

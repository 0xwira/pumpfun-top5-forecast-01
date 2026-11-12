# pumpfun-top5-forecast-01 — entry program (built 2026-11-12, ~14:00 WIB)

`main.py` — Python stdlib only. Reads `/input/task.json` + `/input/history.ndjson`,
writes `/output/forecast.json` (`python3 /bot/main.py` in the grader sandbox).

Model: per-coin log10-cap random walk with damped recent momentum (slope x0.3),
per-coin volatility from recent 10-min moves, a dump-regime mixture, Monte Carlo
(4,000 sims, seeded) projected to each of the 7 answer reads (horizons read from
task.json). Place probabilities = simulated rank frequencies; sizes = simulated
caps at each place (empirical + normal smoothing). A large share of each place is
left unnamed for coins born after the cut: 0.60 base + 0.03/hour of mean horizon,
capped 0.72 (at the final's ~10h mean horizon that is ~0.72 unnamed).

Backtests (local scorer implementing RULES.txt; pseudo-answers = later marks):
- train marks 1-3, answer marks 4-6: mine -43.29 vs baseline -49.96 vs even -45.06
- train marks 1-2, answer marks 3-6: mine -44.03 vs baseline -59.17 vs even -45.06
- train marks 1-4, answer marks 5-6: mine -31.02 vs baseline -33.48 vs even -45.06
House baseline's published practice score is -40.41 (full 6-mark train). Caveats:
backtests are short-horizon (10-40 min) while the final horizon is 7-13h; the
house-model bar is unknown; 625+ entrants. Full dev-history run: 4.1s, valid output.

Status 2026-11-12 ~14:00 WIB: program built + validated locally. NOT submitted.
Submission needs the user's explicit go-ahead: create repo (e.g. 0xwira/pumpfun-top5-
forecast-01), put main.py at root, bundle the RAW session JSONL (news-summary lesson:
self-written exports fail thinking_priced; raw session bundle passes). Deadline
2026-11-12 21:00 WIB (14:00 UTC). No Dune key, no paid key, no on-chain tx needed.

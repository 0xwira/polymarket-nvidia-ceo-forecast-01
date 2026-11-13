# NVIDIA CEO Forecast

Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`: forecast the
Yes price of every market in Polymarket's "Next CEO of NVIDIA?" event,
minute by minute for the 240 minutes after each checkpoint, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per minute.

## How it runs

`main.py` (Python 3, standard library only) reads every `*.in` case
file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<case>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):
`{"forecasts": [{"id": ..., "q": [[...5 numbers...] x 240 rows]}]}`.
It runs fully offline — everything it knows is the 1-minute price
history inside the case file — and it is deterministic: no randomness,
no clock, so two runs of the same case produce identical bytes.

## The model

For each market, two members look at the last 1,440 prices (24 hours):

1. **Robust random walk (weight 0.65).** The median stays at the last
   price; the band widens as `sigma * sqrt(h)` with a per-market
   `sigma = max(1.4826 * median(|1-minute changes|), 0.0002)`. The two
   outer quantiles get an extra ×3.0 stretch, because these prices
   move in jumps and a plain normal band is too confident in the
   tails.
2. **The market's own record (weight 0.35).** The empirical quantiles
   of historical h-minute price changes, read at anchor horizons
   (1, 2, 3, 5, 8, …, 240) and interpolated in `sqrt(h)`, recentered
   so the median sits on the last price. This carries the real shape
   of the moves: long stillness, sudden spikes.

Each member is clamped into [0, 1] before blending. After the
weighted blend, the outer quantiles are widened a further ×1.15 away
from the blended median; rows are then clamped, rounded to 6 decimals
and forced non-decreasing.

## Backtest

On the competition's six practice cases, scored with the house's mean
pinball loss against the prices Polymarket actually printed, the
blend scores **≈ 0.290**, versus the house baseline's **0.363589** —
the bar an entry must beat to take a prize.

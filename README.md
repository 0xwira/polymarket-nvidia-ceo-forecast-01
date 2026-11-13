# NVIDIA CEO Forecast

Entry for make.fun polymarket-nvidia-ceo-forecast-01: forecast the Yes
price of every market of Polymarket's "Next CEO of NVIDIA?" event, one
minute at a time for the 4 hours after each checkpoint, as five
quantiles per minute.

## The model

`main.py` (Python 3, standard library only, fully offline and
deterministic) blends two views of each market's last 1,440 minutes of
1-minute prices:

- **A robust random walk (weight 0.65).** The median stays on the last
  price; the spread grows as sigma * sqrt(h) with a per-market sigma
  from the median absolute 1-minute move, and the outer quantiles are
  stretched threefold because these prices move in jumps.
- **The market's own record (weight 0.35).** Empirical quantiles of
  its historical h-minute price changes at anchor horizons,
  interpolated in sqrt(h) and recentered on the last price.

The blend's outer quantiles are widened a further 15%. Mean reversion
and momentum overlays were tried and rejected: on the practice cases
they only cost loss.

## Backtest

Scored locally against the prices Polymarket actually printed after
the six practice checkpoints, with the competition's pinball loss:

- This model: **0.290**
- House baseline (persistence, sigma 0.0025 for every market): 0.364

The local harness reproduces the baseline's official practice loss
(0.363589) exactly, so the comparison is apples to apples.

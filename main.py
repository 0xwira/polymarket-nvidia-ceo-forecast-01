#!/usr/bin/env python3
"""Entry for make.fun polymarket-nvidia-ceo-forecast-01.

For every market of Polymarket's "Next CEO of NVIDIA?" event this
program forecasts the Yes price at each of the 240 minutes after a
checkpoint, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95). It runs
offline inside the grader's sandbox: everything it knows about a
market is the 1-minute price history inside the case file.

The forecast blends two views of each market's last day of trading
(the most recent 1,440 one-minute prices):

- A robust random walk (weight 0.65). The median stays pinned to the
  last price and the band widens like sigma * sqrt(h), where sigma is
  estimated per market from the median absolute 1-minute move (scaled
  by 1.4826 so it reads like a standard deviation, floored at 0.0002
  so a frozen market still gets a hair of width). The two outer
  quantiles are stretched threefold on top of that: these prices move
  in jumps, not smooth steps, so a plain normal band is far too
  confident about the tails.

- The market's own empirical record (weight 0.35). For a set of anchor
  horizons, the actual distribution of h-minute price changes over the
  window is read off directly, interpolated in sqrt(h) between
  anchors, and recentered so its median sits on the last price. This
  carries the real shape of the moves - long stillness, sudden spikes
  - that no bell curve reproduces.

Both members are clamped into [0, 1] before blending (a price cannot
leave that range, so a member's overshoot is truncated where it lands
rather than averaged back). The blended outer quantiles are then
widened a further 15% away from the blended median, every row is
clamped, rounded to 6 decimals and forced non-decreasing.

Backtested on the competition's six practice cases with the house's
pinball-loss scorer, this blend scores about 0.290 against the house
baseline's 0.363589. Standard library only, no randomness, no clock:
the same case always produces the same bytes.
"""

import json
import math
import os
import sys

WINDOW = 1440            # minutes of history each member looks at
WALK_WEIGHT = 0.65
RECORD_WEIGHT = 0.35
TAIL_STRETCH = 3.0       # outer-quantile stretch inside the walk member
TAIL_POST_SCALE = 1.15   # extra outer widening applied to the blend
SIGMA_FLOOR = 0.0002
MAD_TO_SIGMA = 1.4826

# The five scored levels and the standard normal's quantiles at them
# (the same tabulated values the house baseline uses).
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OF_TAU = {
    0.05: -1.644854,
    0.25: -0.674490,
    0.5: 0.0,
    0.75: 0.674490,
    0.95: 1.644854,
}
# Horizons at which the empirical member reads the change distribution
# directly; between them it interpolates in sqrt(h).
ANCHOR_H = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 160, 200, 240)


def z_at(tau):
    """Normal quantile for a scored level (nearest tabulated fallback)."""
    if tau in Z_OF_TAU:
        return Z_OF_TAU[tau]
    nearest = min(Z_OF_TAU, key=lambda t: abs(t - tau))
    return Z_OF_TAU[nearest]


def clamp6(value):
    """Clamp a price forecast into [0, 1] and round to 6 decimals."""
    return round(min(1.0, max(0.0, value)), 6) + 0.0


def emp_quantile(ordered, q):
    """Quantile of pre-sorted values by linear interpolation.

    Uses the pos = q * (n - 1) convention: q=0 is the minimum, q=1 the
    maximum, and fractional positions blend the two neighbours.
    """
    n = len(ordered)
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def walk_rows(prices, horizon, taus):
    """Robust random walk around the last price.

    sigma comes from the median absolute 1-minute change over the
    window - a median, so the occasional news jump does not blow the
    whole band up - and the outer quantiles are stretched by
    TAIL_STRETCH before clamping.
    """
    window = prices[-WINDOW:]
    steps = sorted(abs(window[i + 1] - window[i]) for i in range(len(window) - 1))
    mad = steps[len(steps) // 2] if steps else 0.0
    sigma = max(MAD_TO_SIGMA * mad, SIGMA_FLOOR)
    x = prices[-1]
    zs = [z_at(t) for t in taus]
    last = len(zs) - 1
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = []
        for i, z in enumerate(zs):
            stretch = TAIL_STRETCH if i in (0, last) else 1.0
            row.append(clamp6(x + z * spread * stretch))
        rows.append(row)
    return rows


def record_rows(prices, horizon, taus):
    """Quantiles read off the market's own history of price changes.

    For each anchor horizon, every h-minute change inside the window
    (prices[i + h] - prices[i]) is collected and its quantiles at the
    scored levels are read by linear interpolation. Horizons above the
    window's length fall back to the single full-span change. Between
    anchors the quantiles are interpolated in sqrt(h), matching the
    way price uncertainty actually accumulates.

    The distribution of changes is then recentered so its median sits
    exactly on the last price x: value = x + (q_tau - q_median). The
    recentering matters because the historical changes are not
    required to be centred on zero - a market that drifted up all day
    would otherwise drag every forecast up with it - while these
    markets are sticky enough that "no move" is the right middle.
    Each row is clamped into [0, 1] here, before any blending.
    """
    window = prices[-WINDOW:]
    n = len(window)
    x = window[-1]
    anchors = [h for h in ANCHOR_H if h <= horizon]
    if horizon not in anchors:
        anchors.append(horizon)
    at_anchor = {}
    for hh in anchors:
        if hh >= n:
            diffs = [window[-1] - window[0]]
        else:
            diffs = sorted(window[i + hh] - window[i] for i in range(n - hh))
        at_anchor[hh] = [emp_quantile(diffs, t) for t in taus]
    mid = len(taus) // 2
    rows = []
    for h in range(1, horizon + 1):
        if h in at_anchor:
            eq = at_anchor[h]
        else:
            lo = max(a for a in anchors if a < h)
            hi = min(a for a in anchors if a > h)
            f = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            eq = [at_anchor[lo][i] + (at_anchor[hi][i] - at_anchor[lo][i]) * f
                  for i in range(len(taus))]
        rows.append([clamp6(x + (eq[i] - eq[mid])) for i in range(len(taus))])
    return rows


def forecast_market(market, horizon, taus):
    """Blend the two members into the final quantile rows.

    A straight weighted mean per quantile (0.65 walk, 0.35 record),
    then one last tail adjustment: the two outer quantiles are pushed
    15% further away from the blended median, because even the blend
    underestimates how often these markets jump several ticks at once.
    The median itself is never moved by that push. Finally each row is
    clamped to [0, 1], rounded, and swept left to right so the five
    quantiles can never come out crossed.
    """
    prices = market["prices"]
    walk = walk_rows(prices, horizon, taus)
    record = record_rows(prices, horizon, taus)
    mid = len(taus) // 2
    last = len(taus) - 1
    rows = []
    for h in range(horizon):
        blended = [WALK_WEIGHT * walk[h][i] + RECORD_WEIGHT * record[h][i]
                   for i in range(len(taus))]
        median = blended[mid]
        row = []
        for i, v in enumerate(blended):
            if i in (0, last):
                v = median + (v - median) * TAIL_POST_SCALE
            row.append(clamp6(v))
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def run_case(path, out_dir):
    with open(path, encoding="utf-8") as fh:
        case = json.load(fh)
    horizon = case["horizon_minutes"]
    taus = case["quantiles"]
    forecasts = [
        {"id": m["id"], "q": forecast_market(m, horizon, taus)}
        for m in case["markets"]
    ]
    target = os.path.join(out_dir, case["case"] + ".out")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump({"forecasts": forecasts}, fh, separators=(",", ":"))
    return target


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print(f"no case files in {in_dir}", file=sys.stderr)
        return 1
    for name in names:
        print(f"wrote {run_case(os.path.join(in_dir, name), out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

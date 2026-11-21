#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy, five quantiles of its Exophase earn rate at
the case's target, as a percent.  Standard library only, no network,
deterministic: the same case always yields the same bytes.

Model
-----
A trophy's rate is holders / players_tracked, and Exophase refreshes both
counts in jumps, not continuously (launch night: the page sat unchanged
at 638 players tracked for hours, then jumped to 3,868 with every rate
but the 100% one down -- dilution: newly tracked players hold far fewer
trophies).  So instead of extrapolating the rate itself, this program
forecasts its two parts separately and divides:

  * players_tracked at the target: Exophase refreshes arrive as a jump
    process.  The refresh rate is a Gamma posterior over the jumps seen
    in the case history (prior mean about one refresh every 1.6 hours:
    at launch the tracked base is a tiny fraction of the launch player
    base, so a refresh is expected with substantial probability inside
    even a 90-minute window, and any refresh must add a large multiple
    of players -- the launch-night jump was x6.06 within five hours of
    the first probe), and each refresh multiplies the tracked base by a
    factor whose log-mean is estimated from the jumps seen (prior
    median x4).  Projected log growth is capped, so a launch-night
    explosion never compounds into an absurd week-one player count.

  * holders of each trophy at the target: the current holders, grown by
    a per-trophy power law in time since release fitted on the distinct
    reads (players keep earning trophies), plus the trophies the newly
    tracked players bring with them.  A newly tracked player holds
    trophy i with probability p_i, estimated from the observed refreshes
    (holders gained / players gained at each jump, shrunk toward a
    power-law prior p_i = rate_i ** gamma, gamma ~ 1.6: rare trophies
    are held by a much smaller share of newcomers than common ones, and
    a trophy at 100% -- held by every tracked player so far -- is held
    by newcomers too, so its rate stays near 100% instead of diluting).

  * median rate = forecast holders / forecast players_tracked.

Spread
------
The spread on the logit scale combines the compound-Poisson variance
of the refresh jumps still to come (each future refresh moves a
trophy's logit by roughly ln((1 + kappa*(f-1)) / f)), the uncertainty
in the holders' growth and in the player-count projection, and a floor
for refresh timing and page rounding.  When the history is flat the
spread is wide on purpose: a refresh the case has never seen is the
dominant risk, and a narrow band pays at every trophy it misses.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- priors and caps (fixed in advance; see the docstring) -----------------
LAM_A0 = 1.5        # Gamma prior: pseudo refresh jumps observed
LAM_B0 = 0.10       # Gamma prior: pseudo days of history (prior mean 15/day)
F_MED_PRIOR = 4.0   # prior median players-tracked multiplier per refresh
GAMMA_PRIOR = 1.6   # new-player hold-rate power law: p = rate ** gamma
KAPPA_MIN = 0.02    # new players hold at least this share of a trophy's rate
LOG_R_CAP = 2.0     # max projected log growth of players_tracked (x7.4)
GROWTH_CAP = 50.0   # max projected holders growth factor
B_MAX = 2.0         # max holders power-law exponent
B_MIN = -0.5        # min holders power-law exponent
SPREAD_FLOOR = 0.40
SPREAD_CAP = 2.5
DELTA_VAR = 0.09    # extra variance of one refresh's logit shift (sd 0.3)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_pct(x):
    return clamp(x, 0.01, 99.99)


def logit(pct):
    c = clamp_pct(pct)
    return math.log(c / (100.0 - c))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-clamp(l, -20.0, 20.0)))


def round4(x):
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_time(s):
    """ISO 8601 timestamp -> epoch seconds (float)."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ols_slope(xs, ys):
    """Least-squares slope of y on x; 0.0 when x does not vary."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return 0.0
    return num / den


def distinct_states(history):
    """Collapse consecutive identical reads; one entry per Exophase state.

    Returns a list of (t_seconds, players_tracked, rates) oldest first.
    A read that changes nothing carries no information beyond the last.
    """
    states = []
    prev = None
    for h in history:
        rates = h.get("rates") or []
        n = h.get("players_tracked")
        key = (n, tuple(rates))
        if key == prev:
            continue
        prev = key
        states.append((parse_time(h.get("at", "")), n, rates))
    return states


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    if not trophies:
        return {"trophies": []}
    if not history:
        # No reads at all: a flat 1% median with a very wide band.
        rows = []
        for t in trophies:
            q = [round4(inv_logit(logit(1.0) + z * 2.0)) for z in Z]
            rows.append({"id": t["id"], "q": q})
        return {"trophies": rows}

    states = distinct_states(history)
    n_last = states[-1][1] or 0
    last_rates = states[-1][2]
    # The exposure time for the refresh-rate posterior is the full span of
    # reads the house made, flat ones included: no refresh was seen in
    # that whole window, and that is the evidence about the refresh rate.
    t_first = parse_time(history[0].get("at", ""))
    t_last = parse_time(history[-1].get("at", ""))
    span_days = max((t_last - t_first) / 86400.0, 0.0)

    # Time since release, in days, floored so log-time stays finite.
    rel = case.get("game") or {}
    t0 = parse_time(str(rel.get("release_date", "")) + "T00:00:00Z") \
        if rel.get("release_date") else t_first
    if t0 <= 0:
        t0 = t_first
    tau = [max((t - t0) / 86400.0, 0.05) for (t, _, _) in states]
    tau_last = tau[-1]
    tau_target = tau_last + horizon_days

    # --- refresh jumps observed in the history ---------------------------
    jumps = []  # (index into states, dN, ln f)
    for j in range(1, len(states)):
        n_prev = states[j - 1][1] or 0
        n_now = states[j][1] or 0
        if n_prev > 0 and n_now > n_prev:
            jumps.append((j, n_now - n_prev, math.log(n_now / n_prev)))
    n_jumps = len(jumps)

    lam = (LAM_A0 + n_jumps) / (LAM_B0 + span_days)  # refreshes per day
    if jumps:
        ln_f_obs = [lf for (_, _, lf) in jumps]
        w = n_jumps / (n_jumps + 2.0)
        ln_f = w * (sum(ln_f_obs) / len(ln_f_obs)) + (1.0 - w) * math.log(F_MED_PRIOR)
    else:
        ln_f = math.log(F_MED_PRIOR)
    f_med = math.exp(ln_f)

    # Projected players_tracked at the target.
    #
    # When the history shows real growth, project the RECENT log growth
    # rate (last day of reads), damped over the horizon -- launch growth
    # does not compound for a week -- instead of compounding the jump
    # process, which would explode once refreshes are frequent but
    # small.  When the history is flat or nearly so, there is no recent
    # growth rate to project, so fall back to the jump-process prior:
    # the Gamma refresh rate times the expected log multiplier.
    log_r = None
    if n_jumps >= 2 and span_days > 0:
        window = min(span_days, 1.0)
        t_cut = t_last - window * 86400.0
        win = [(t, n) for (t, n, _) in states if t >= t_cut and n and n > 0]
        if len(win) >= 2 and win[0][1] > 0 and win[-1][1] > win[0][1]:
            r_n = math.log(win[-1][1] / win[0][1]) / ((win[-1][0] - win[0][0]) / 86400.0)
            log_r = clamp(r_n * horizon_days * math.exp(-horizon_days / 3.0), 0.0, LOG_R_CAP)
    if log_r is None:
        log_r = clamp(lam * horizon_days * ln_f, 0.0, LOG_R_CAP)
    n_target = n_last * math.exp(log_r) if n_last > 0 else 0.0
    d_n = max(n_target - n_last, 0.0)
    jumps_expected = lam * horizon_days

    # --- per-trophy new-player hold rates, from the observed jumps -------
    # Fit the power-law exponent gamma from jumps with positive gains.
    gamma = GAMMA_PRIOR
    gx, gy = [], []
    for (j, d_n_j, _) in jumps:
        rates_prev = states[j - 1][2]
        rates_now = states[j][2]
        n_prev = states[j - 1][1] or 0
        n_now = states[j][1] or 0
        if d_n_j <= 0 or n_prev <= 0:
            continue
        for i in range(len(trophies)):
            r_prev = rates_prev[i] if i < len(rates_prev) else None
            r_now = rates_now[i] if i < len(rates_now) else None
            if r_prev is None or r_now is None or r_prev <= 0.05:
                continue
            h_prev = r_prev / 100.0 * n_prev
            h_now = r_now / 100.0 * n_now
            p = clamp((h_now - h_prev) / d_n_j, 0.0, 1.0) if h_now > h_prev else 0.0
            if p > 0.0005:
                gx.append(math.log(r_prev / 100.0))
                gy.append(math.log(p))
    if len(gx) >= 5:
        g_fit = ols_slope(gx, gy)
        if 1.0 <= g_fit <= 2.5:
            gamma = g_fit

    # --- per-trophy holders growth (power law in time since release) -----
    # Fit log holders on log tau per trophy; shrink toward the pooled slope.
    # Keep the intercept too: when the last read rounds a rare trophy to
    # 0.00%, the fitted trend still says how many holders there are.
    b_pool_num = []
    b_per = []
    a_per = []
    for i in range(len(trophies)):
        xs, ys = [], []
        for s_idx, (_, n_s, rates_s) in enumerate(states):
            r = rates_s[i] if i < len(rates_s) else None
            if r is None or n_s is None or n_s <= 0 or r <= 0:
                continue
            xs.append(math.log(tau[s_idx]))
            ys.append(math.log(r / 100.0 * n_s))
        if len(xs) >= 3:
            b = ols_slope(xs, ys)
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            a = my - b * mx
        else:
            b, a = 0.0, 0.0
        b_per.append(clamp(b, B_MIN, B_MAX))
        a_per.append(a)
        b_pool_num.append(b_per[-1])
    b_pool = median(b_pool_num) if b_pool_num else 0.0

    rows = []
    for i, t in enumerate(trophies):
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            r_last = 0.0
        rho = clamp_pct(r_last) / 100.0
        l0 = logit(r_last)
        h_raw = rho * n_last if n_last > 0 else 0.0
        # Holders implied by the fitted trend at the last read.  When the
        # page rounds a rare trophy to 0.00% the raw count is 0, but the
        # trend (fitted on earlier, non-zero reads) still knows the
        # holders are there; holders never fall, so take the larger.
        h_fit_last = 0.0
        if a_per[i] != 0.0 or b_per[i] != 0.0:
            h_fit_last = math.exp(a_per[i] + b_per[i] * math.log(tau_last))
        h_last = max(h_raw, h_fit_last) if h_raw <= 0 else h_raw

        # kappa_i: share of the trophy's rate that a newly tracked player
        # holds.  Data first (median over jumps), prior as the fallback.
        k_data = []
        for (j, d_n_j, _) in jumps:
            rates_prev = states[j - 1][2]
            rates_now = states[j][2]
            n_prev = states[j - 1][1] or 0
            n_now = states[j][1] or 0
            if d_n_j <= 0 or n_prev <= 0:
                continue
            r_prev = rates_prev[i] if i < len(rates_prev) else None
            r_now = rates_now[i] if i < len(rates_now) else None
            if r_prev is None or r_now is None or r_prev <= 0.05:
                continue
            h_prev = r_prev / 100.0 * n_prev
            h_now = r_now / 100.0 * n_now
            p = clamp((h_now - h_prev) / d_n_j, 0.0, 1.0) if h_now > h_prev else 0.0
            k_data.append(clamp(p / (r_prev / 100.0), 0.0, 1.5))
        k_prior = clamp(rho ** (gamma - 1.0), KAPPA_MIN, 1.0) if rho > 0 else KAPPA_MIN
        if k_data:
            w = n_jumps / (n_jumps + 2.0)
            kappa = w * median(k_data) + (1.0 - w) * k_prior
        else:
            kappa = k_prior
        p_new = clamp(kappa * rho, 0.0, 1.0)

        # Holders growth of the current cohort over the horizon.
        w_b = len(states) / (len(states) + 4.0)
        b_i = w_b * b_per[i] + (1.0 - w_b) * b_pool
        growth = clamp((tau_target / tau_last) ** b_i, 1.0 / GROWTH_CAP, GROWTH_CAP) \
            if tau_last > 0 else 1.0
        h_target = h_last * growth + p_new * d_n
        if n_target > 0:
            h_target = clamp(h_target, 0.0, n_target)
            rate_target = 100.0 * h_target / n_target
        else:
            rate_target = r_last
        m = logit(rate_target) - l0

        # Spread: future-refresh variance, holders-growth and player-count
        # uncertainty, plus a floor.  A trophy newcomers all hold (kappa
        # near 1) barely moves at a refresh; a rare one moves a lot.
        delta = math.log((1.0 + kappa * (f_med - 1.0)) / f_med) if f_med > 1 else 0.0
        s_jump2 = jumps_expected * (delta * delta + DELTA_VAR)
        s_prog2 = (0.25 * abs(math.log(growth)) + 0.10) ** 2
        s_n2 = (0.30 * log_r) ** 2
        s = math.sqrt(s_jump2 + s_prog2 + s_n2 + SPREAD_FLOOR ** 2)
        s = clamp(s, 0.30, SPREAD_CAP)

        q = []
        prev_q = -1.0
        for z in Z:
            v = round4(clamp_pct(inv_logit(l0 + m + z * s)))
            if v < prev_q:
                v = prev_q
            q.append(v)
            prev_q = v
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(n for n in os.listdir(INPUT_DIR) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, name[:-3] + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

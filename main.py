#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry (stdlib only).

Reads /input/task.json + /input/history.ndjson (10-min marks of pump.fun
coins launched since t0, ranked by market cap), writes /output/forecast.json.

Model: per-coin log10-cap random walk with damped recent momentum drift,
per-coin volatility from recent moves, a dump-regime mixture, and Monte
Carlo projection to each of the seven answer reads (horizons read from
task.json, never hard-coded). Place probabilities come from simulated
rank frequencies; size bins from the simulated caps at each place.
A share of every place is deliberately left unnamed for coins born after
the cut (the practice notes 3 of 9 top-5 holders were born after the cut),
growing with forecast horizon.
"""
import json, math, os, random, sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
RNG = random.Random(20261112)
# 12k sims: the practice-input run takes ~12s against a 300s cap, and the
# extra draws cut Monte Carlo noise in the small place probabilities.
N_SIMS = 12000


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log10_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = task["places"]
    edges = task["size"]["edges"]
    reads = [parse_time(r) for r in task["reads"]]

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append((parse_time(m["at"]), m["rows"]))
    if not marks:
        raise SystemExit("no ok marks in history")
    marks.sort(key=lambda t: t[0])
    last_t = marks[-1][0]
    horizons = [max(0.0, (r - last_t).total_seconds() / 3600.0) for r in reads]
    mean_h = sum(horizons) / len(horizons)

    # --- per-coin series: mint -> list of (t_hours_from_last, log10 cap) ---
    series = {}
    created = {}
    for t, rows in marks:
        dt = (t - last_t).total_seconds() / 3600.0
        for r in rows:
            mint = r["mint"]
            series.setdefault(mint, []).append((dt, log10_sol(r["market_cap_lamports"])))
            created.setdefault(mint, r.get("created_at"))
    last_chart = sorted(marks[-1][1], key=lambda r: r["rank"])
    last_rank = {r["mint"]: r["rank"] for r in last_chart}

    # candidate set: last chart top-100 + any coin top-20 in the last 3 marks
    cand = set(r["mint"] for r in last_chart[:100])
    for _, rows in marks[-3:]:
        for r in sorted(rows, key=lambda x: x["rank"])[:20]:
            cand.add(r["mint"])
    cand = sorted(m for m in cand if m in series)  # sorted: deterministic sims

    # --- per-coin drift (damped momentum) and volatility ---
    params = {}
    for mint in cand:
        pts = sorted(series[mint])
        xs = [x for _, x in pts]
        last_x = xs[-1]
        # weighted least-squares slope over the series (recent-weighted)
        if len(pts) >= 2:
            ts = [t for t, _ in pts]
            w = [0.6 ** (len(pts) - 1 - i) for i in range(len(pts))]
            sw = sum(w)
            mt = sum(wi * ti for wi, ti in zip(w, ts)) / sw
            mx = sum(wi * xi for wi, xi in zip(w, xs)) / sw
            den = sum(wi * (ti - mt) ** 2 for wi, ti in zip(w, ts))
            slope = sum(wi * (ti - mt) * (xi - mx) for wi, ti, xi in zip(w, ts, xs)) / den if den > 1e-9 else 0.0
        else:
            slope = 0.0
        # damp momentum hard: memecoin leaders mean-revert; keep half the slope
        slope = max(-2.0, min(2.0, slope)) * float(os.environ.get("SLOPE_DAMP", "0.3"))
        # volatility: sd of 10-min log steps, annualised to per-hour
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        if len(diffs) >= 2:
            md = sum(diffs) / len(diffs)
            vol10 = math.sqrt(sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1))
        else:
            vol10 = 0.25
        vol_h = max(0.20, min(0.90, vol10 * math.sqrt(6.0)))
        params[mint] = (last_x, slope, vol_h)

    # --- Monte Carlo projection ---
    # place_counts[read][place] = {mint: count}; place_caps[read][place] = [log10 caps]
    place_counts = [{k: {} for k in range(1, places + 1)} for _ in reads]
    place_caps = [{k: [] for k in range(1, places + 1)} for _ in reads]
    for _ in range(N_SIMS):
        # one drift draw per coin per sim (shared across reads), then per-read noise
        proj = {}
        for mint in cand:
            last_x, slope, vol_h = params[mint]
            drift = RNG.gauss(slope, 0.25)
            proj[mint] = (last_x, drift, vol_h)
        for ri, h in enumerate(horizons):
            scored = []
            for mint in cand:
                last_x, drift, vol_h = proj[mint]
                x = last_x + drift * h + RNG.gauss(0.0, vol_h * math.sqrt(h + 0.25))
                # dump regime: with prob growing in h, the coin craters ~1 dex
                if RNG.random() < 0.10 * (1.0 - math.exp(-h / 3.0)) + (0.05 if h > 2 else 0.0):
                    x -= 1.0
                scored.append((x, mint))
            scored.sort(key=lambda t: (-t[0], t[1]))
            for k in range(1, places + 1):
                x, mint = scored[k - 1]
                place_counts[ri][k][mint] = place_counts[ri][k].get(mint, 0) + 1
                place_caps[ri][k].append(x)
    # average frequencies / caps across the seven reads (one forecast, scored at each)
    probabilities = []
    sizes = []
    # unnamed share per place: newcomers take more places the further out we look
    unnamed = max(0.35, min(0.72, float(os.environ.get("UNNAMED_BASE", "0.60")) + 0.03 * mean_h))
    named_mass = 1.0 - unnamed
    for k in range(1, places + 1):
        freq = {}
        for ri in range(len(reads)):
            for mint, c in place_counts[ri][k].items():
                freq[mint] = freq.get(mint, 0) + c / N_SIMS
        for mint in freq:
            freq[mint] /= len(reads)
        # keep the strongest names, drop dust (p < 0.002 after scaling)
        scaled = sorted(((m, p * named_mass) for m, p in freq.items()),
                        key=lambda t: -t[1])
        total = 0.0
        for mint, p in scaled:
            if p < 0.002 or total + p > named_mass:
                continue
            probabilities.append({"mint": mint, "place": k,
                                  "p": math.floor(p * 1e6) / 1e6})
            total += math.floor(p * 1e6) / 1e6
        # sizes: empirical sim distribution per place + normal smoothing
        caps = []
        for ri in range(len(reads)):
            caps.extend(place_caps[ri][k])
        caps.sort()
        n = len(caps)
        mean = sum(caps) / n
        sd = max(float(os.environ.get("SIZE_SD_FLOOR", "0.45")), math.sqrt(sum((c - mean) ** 2 for c in caps) / n))
        sd = min(sd, 0.90)
        for b in range(len(edges) + 1):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            emp = sum(1 for c in caps if lo <= c < hi) / n if b < len(edges) else sum(1 for c in caps if c >= lo) / n
            if b == 0:
                emp = sum(1 for c in caps if c < hi) / n
            nor = norm_cdf((hi - mean) / sd) - norm_cdf((lo - mean) / sd) if math.isfinite(lo) and math.isfinite(hi) else (
                norm_cdf((hi - mean) / sd) if math.isinf(lo) else 1.0 - norm_cdf((lo - mean) / sd))
            p = 0.65 * emp + 0.35 * nor
            p = math.floor(p * 1e6) / 1e6
            if p > 0:
                sizes.append({"place": k, "bin": b, "p": p})
    # top5: best-frequency coin per place, greedy dedup in place order
    top5 = []
    used = set()
    for k in range(1, places + 1):
        freq = {}
        for ri in range(len(reads)):
            for mint, c in place_counts[ri][k].items():
                freq[mint] = freq.get(mint, 0) + c
        for mint, _ in sorted(freq.items(), key=lambda t: -t[1]):
            if mint not in used:
                top5.append(mint)
                used.add(mint)
                break
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps({"probabilities": probabilities, "top5": top5,
                            "sizes": sizes}) + "\n")
    print(f"wrote {task['output']}: {len(probabilities)} prob rows, "
          f"{len(sizes)} size rows, top5={top5}", file=sys.stderr)


if __name__ == "__main__":
    main()

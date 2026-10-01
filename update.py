"""Falling Knives daily updater.

Runs in GitHub Actions after the US market close. Pulls daily prices for every
S&P 500 member from Yahoo Finance (via yfinance), ranks the 50 worst performers
month-to-date, rebuilds the blue-chip section, and writes data.js for the site.

Reasons come from reasons.json, which this script only READS. Stocks without a
reason are shown as "Reason pending" with their latest headlines.
"""
import calendar
import datetime as dt
import io
import json
import math
import re
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

CONSTITUENTS = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
SKIP_BIG = {"GOOG", "FOX", "NWS"}          # second share classes
TOP_N, BIG_N = 50, 50
GAIN_N, VOL_N = 50, 50


def r2(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), 2)


def week_key(d):
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def load_universe():
    df = pd.read_csv(io.StringIO(requests.get(CONSTITUENTS, timeout=30).text))
    df["Y"] = df["Symbol"].str.replace(".", "-", regex=False)
    return {r.Y: {"name": r.Security, "sector": r["GICS Sector"], "sub": r["GICS Sub-Industry"] if isinstance(r.get("GICS Sub-Industry"), str) else ""} for _, r in df.iterrows()}


def is_tech(sector, sub):
    return sector == "Information Technology" or bool(re.search(r"Semiconductor|Interactive Media", sub or ""))


def pick_pattern(r, sector, sub, secmed, spx_mtd):
    """The four traits the best knives shared in the 2021-2026 backtest."""
    sm = secmed.get(sector)
    t = is_tech(sector, sub)
    c = (sm is not None and sm <= -3) or (spx_mtd is not None and spx_mtd <= -3)
    n = sm is not None and r["mtd"] - sm > -12
    d = r.get("dd52") is not None and r["dd52"] <= -30
    return {"t": t, "c": bool(c), "n": bool(n), "d": bool(d), "score": int(t) + int(bool(c)) + int(bool(n)) + int(bool(d)),
            "sm": r2(sm) if sm is not None else None}


def download(symbols, period="2y", min_len=61):
    frames = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=period, interval="1d", auto_adjust=False,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # noqa
                raw = None
                print("download retry", e, file=sys.stderr); time.sleep(5)
        if raw is None or len(raw) == 0:
            print(f"download failed for chunk starting {chunk[0]}", file=sys.stderr)
            continue
        for s in chunk:
            try:
                sub = raw[s] if len(chunk) > 1 else raw
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(sub) >= min_len:
                    frames[s] = sub
            except Exception:
                pass
    return frames


def ny_now():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # noqa
        return dt.datetime.utcnow() - dt.timedelta(hours=4)


def expected_session(now=None):
    """Most recent trading day whose 4pm close has already happened (with a 15-minute buffer)."""
    now = now or ny_now()
    d = now.date()
    if not is_trading_day(d) or now.hour * 60 + now.minute < 16 * 60 + 15:
        d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def _upto(f, d):
    return f[[x.date() <= d for x in f.index]]


def align_frames(frames, syms, now=None):
    """Make every symbol end on the same completed session.

    Yahoo often serves the newest daily bar for only some symbols in the first hours after the
    close, and serves a partial bar during market hours. Previously one early symbol could set the
    date and knock out the other 500 (run fails), or every symbol lagged a day (stale site).
    Here: drop partial bars, re-fetch symbols missing the latest session, and only then pick the date.
    """
    exp = expected_session(now)
    frames = {s: _upto(f, exp) for s, f in frames.items()}
    frames = {s: f for s, f in frames.items() if len(f)}
    missing = [s for s in syms if s not in frames]
    if missing:
        print(f"re-downloading {len(missing)} symbols with no data", file=sys.stderr)
        time.sleep(5)
        for s, f in download(missing).items():
            f = _upto(f, exp)
            if len(f): frames[s] = f
    for attempt in range(3):
        stale = [s for s in syms if s in frames and frames[s].index[-1].date() < exp]
        if not stale:
            break
        print(f"{len(stale)} symbols missing the {exp} bar; refetching (try {attempt + 1})", file=sys.stderr)
        time.sleep(10 * (attempt + 1))
        for s, f in download(stale, period="5d", min_len=1).items():
            f = _upto(f, exp)
            if not len(f): continue
            c = pd.concat([frames[s], f])
            frames[s] = c[~c.index.duplicated(keep="last")].sort_index()
    ends = {}
    for s in syms:
        if s in frames:
            d = frames[s].index[-1].date(); ends[d] = ends.get(d, 0) + 1
    if not ends:
        sys.exit("No price data downloaded; not publishing.")
    last_date = exp if ends.get(exp, 0) >= 450 else max(ends, key=lambda d: (ends[d], d))
    frames = {s: _upto(f, last_date) for s, f in frames.items()}
    print(f"expected session {exp}; using {last_date} ({ends.get(last_date, 0)} symbols end there)")
    return frames, last_date


def metrics(s, df, month_start):
    dates = [d.date() for d in df.index]
    C = [float(x) for x in df["Close"]]
    H = [float(x) if not pd.isna(x) else c for x, c in zip(df["High"], C)]
    V = [float(x) if not pd.isna(x) else 0.0 for x in df["Volume"]]
    n = len(C)
    last, last_d = C[-1], dates[-1]
    pre = [c for c, d in zip(C, dates) if d < month_start]
    base = pre[-1] if pre else C[0]
    ye = [c for c, d in zip(C, dates) if d.year < last_d.year]
    ytdb = ye[-1] if ye else None
    y1 = list(range(max(0, n - 252), n))
    hi_i = max(y1, key=lambda i: C[i]); lo_i = min(y1, key=lambda i: C[i])
    hi2_i = max(range(n), key=lambda i: C[i])
    m3 = list(range(max(0, n - 63), n)); p3_i = max(m3, key=lambda i: C[i])
    # weeks (Monday-based), last close of each week
    W = []
    for c, d in zip(C, dates):
        k = week_key(d)
        if W and W[-1]["k"] == k:
            W[-1]["c"], W[-1]["e"] = c, d
        else:
            W.append({"k": k, "c": c, "e": d})
    for i in range(1, len(W)):
        W[i]["r"] = W[i]["c"] / W[i - 1]["c"] - 1
    cur_k = week_key(last_d)
    done = [w for w in W if w["k"] != cur_k]
    cur = W[-1]
    last_pos = next((done[i] for i in range(len(done) - 1, 0, -1) if done[i].get("r", 0) > 0), None)
    dw = 0
    for i in range(len(done) - 1, 0, -1):
        if done[i].get("r", 0) < 0: dw += 1
        else: break
    dd = 0
    for i in range(n - 1, 0, -1):
        if C[i] < C[i - 1]: dd += 1
        else: break
    sma = lambda k: sum(C[-k:]) / min(k, n)
    g = l = 0.0
    start = max(1, n - 200)
    for i in range(start, n):
        ch = C[i] - C[i - 1]; G, L = max(ch, 0), max(-ch, 0)
        if i < start + 14: g += G / 14; l += L / 14
        else: g = (g * 13 + G) / 14; l = (l * 13 + L) / 14
    rsi = 100 - 100 / (1 + g / l) if l else 100.0
    vol = lambda k: sum(V[-k:]) / k
    # bottoming signals
    rs_series = []
    g2 = l2 = 0.0
    for i in range(start, n):
        ch = C[i] - C[i - 1]; G, L = max(ch, 0), max(-ch, 0)
        if i < start + 14: g2 += G / 14; l2 += L / 14
        else: g2 = (g2 * 13 + G) / 14; l2 = (l2 * 13 + L) / 14
        rs_series.append(100 - 100 / (1 + g2 / l2) if l2 else 100.0)
    sma50s = [None] * n
    run = 0.0
    for i in range(n):
        run += C[i]
        if i >= 50: run -= C[i - 50]
        if i >= 49: sma50s[i] = run / 50
    worst = None
    for i in range(1, n):
        if dates[i] >= month_start:
            r = C[i] / C[i - 1] - 1
            if worst is None or r < worst[0]: worst = (r, dates[i])
    ret = lambda k: (C[-1] / C[max(0, n - 1 - k)] - 1) * 100
    # volatility: annualized stdev of daily log returns
    lr = [math.log(C[i] / C[i - 1]) for i in range(1, n) if C[i - 1] > 0 and C[i] > 0]
    def sd(x):
        if len(x) < 2: return None
        m = sum(x) / len(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
    v30 = sd(lr[-21:]); v1y = sd(lr[-252:])
    dr = [(dates[i], C[i] / C[i - 1] - 1) for i in range(max(1, n - 21), n)]
    bu = max(dr, key=lambda x: x[1]); bd = min(dr, key=lambda x: x[1])
    uw = 0
    for i in range(len(done) - 1, 0, -1):
        if done[i].get("r", 0) > 0: uw += 1
        else: break
    sig = []
    if n > 60 and sma50s[-1] and C[-1] > sma50s[-1] and any(sma50s[i] and C[i] < sma50s[i] for i in range(n - 10, n - 1)):
        sig.append("Back above 50-day average")
    if len(rs_series) > 12:
        lo = min(rs_series[-10:-1])
        if lo < 30 and rs_series[-1] >= lo + 10:
            sig.append("RSI turning up from oversold")
    if len(done) > 3 and done[-1].get("r", 0) > 0:
        k = 0
        for i in range(len(done) - 2, 0, -1):
            if done[i].get("r", 0) < 0: k += 1
            else: break
        if k >= 2:
            sig.append(f"First up week after {k} down weeks")
    if n > 70:
        lowA, lowB = min(C[-10:]), min(C[-30:-10])
        if lowB == min(C[-63:]) and lowA > lowB * 1.01 and C[-1] > lowA * 1.02:
            sig.append("Higher low")
    last_neg = next((done[i] for i in range(len(done) - 1, 0, -1) if done[i].get("r", 0) < 0), None)
    iso = lambda d: d.isoformat() if d else None
    return {
        "s": s, "last": r2(last), "prev": r2(C[-2]), "day": r2((C[-1] / C[-2] - 1) * 100), "asof": iso(last_d),
        "base": round(base, 4), "mtd": r2((last / base - 1) * 100),
        "w1": r2(ret(5)), "m1": r2(ret(21)), "m3": r2(ret(63)),
        "ytd": r2((last / ytdb - 1) * 100) if ytdb else None, "y1": r2(ret(min(251, n - 1))),
        "hi52": r2(C[hi_i]), "hi52d": iso(dates[hi_i]), "dd52": r2((last / C[hi_i] - 1) * 100),
        "tdSinceHi": n - 1 - hi_i, "hi2y": r2(C[hi2_i]), "hi2yd": iso(dates[hi2_i]),
        "lo52": r2(C[lo_i]), "lo52d": iso(dates[lo_i]), "offLo": r2((last / C[lo_i] - 1) * 100),
        "p3": r2(C[p3_i]), "p3d": iso(dates[p3_i]), "ddp3": r2((last / C[p3_i] - 1) * 100),
        "lastPosWk": iso(last_pos["e"]) if last_pos else None,
        "lastPosWkRet": r2(last_pos["r"] * 100) if last_pos else None,
        "downWks": dw, "curWk": r2(cur.get("r", 0) * 100), "downDays": dd,
        "rsi": r2(rsi), "v50": r2((last / sma(50) - 1) * 100), "v200": r2((last / sma(200) - 1) * 100),
        "volx": r2(vol(5) / vol(50)) if vol(50) else None,
        "worstDay": r2(worst[0] * 100) if worst else None, "worstDayD": iso(worst[1]) if worst else None,
        "vol30": r2(v30 * math.sqrt(252) * 100) if v30 else None,
        "vol1y": r2(v1y * math.sqrt(252) * 100) if v1y else None,
        "adm": r2(sum(abs(x[1]) for x in dr) / len(dr) * 100),
        "bigUp": r2(bu[1] * 100), "bigUpD": iso(bu[0]), "bigDn": r2(bd[1] * 100), "bigDnD": iso(bd[0]),
        "sig": sig, "upWks": uw, "lastNegWk": iso(last_neg["e"]) if last_neg else None,
        "lastNegWkRet": r2(last_neg["r"] * 100) if last_neg else None,
        "_dr": {dates[i].isoformat(): C[i] / C[i - 1] - 1 for i in range(max(1, n - 252), n)},
        "wks": [r2(w.get("r", 0) * 100) for w in W[-10:]],
        "wkends": [w["e"].strftime("%b %-d") for w in W[-10:]],
    }


def market_caps(symbols):
    def one(s):
        for _ in range(2):
            try:
                v = yf.Ticker(s).fast_info.get("marketCap")
                if v: return s, float(v)
            except Exception:
                time.sleep(1)
        return s, None
    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, symbols))


def details(s):
    out = {}
    try:
        t = yf.Ticker(s)
        info = t.info or {}
        out.update({
            "tgt": r2(info.get("targetMeanPrice")), "rec": info.get("recommendationKey"),
            "nA": info.get("numberOfAnalystOpinions"), "ind": info.get("industry"),
            "fpe": r2(info.get("forwardPE")),
            "dy": r2((info.get("trailingAnnualDividendYield") or 0) * 100) or None,
        })
        now = time.time()
        cands = [x for x in (info.get("earningsTimestampStart"), info.get("earningsTimestamp")) if x]
        fut = [x for x in cands if x > now - 86400]
        ts = min(fut) if fut else (max(cands) if cands else None)
        out["ed"] = dt.datetime.utcfromtimestamp(ts).date().isoformat() if ts else None
        out["edEst"] = bool(info.get("isEarningsDateEstimate")) if ts else None
        news = []
        for item in (t.news or [])[:6]:
            c = item.get("content", item)
            title = c.get("title")
            when = c.get("pubDate") or c.get("displayTime")
            if not when and item.get("providerPublishTime"):
                when = dt.datetime.utcfromtimestamp(item["providerPublishTime"]).isoformat()
            if title:
                d = (when or "")[:10]
                try: d = dt.date.fromisoformat(d).strftime("%b %-d")
                except Exception: pass
                news.append([d, title])
        out["news"] = news[:3]
    except Exception as e:
        print("details fail", s, e, file=sys.stderr)
    return s, out


def write_history(syms, frames, uni, M):
    """One small JSON file per stock with ~2 years of daily candles for the chart view."""
    os.makedirs("history", exist_ok=True)
    n = 0
    for s in syms:
        df = frames.get(s)
        if df is None:
            continue
        C = [float(x) for x in df["Close"]]
        pick = lambda col: [r2(c if pd.isna(x) else x) for x, c in zip(df[col], C)]
        O, H, L = pick("Open"), pick("High"), pick("Low")
        out = {
            "s": s, "name": uni.get(s, {}).get("name", s), "sector": uni.get(s, {}).get("sector"),
            "t": [d.strftime("%Y-%m-%d") for d in df.index],
            "o": O, "h": [max(h, o, r2(c)) for h, o, c in zip(H, O, C)],
            "l": [min(l, o, r2(c)) for l, o, c in zip(L, O, C)], "c": [r2(c) for c in C],
            "v": [0 if pd.isna(v) else int(v) for v in df["Volume"]],
        }
        if s in M:
            out["m"] = {k: v for k, v in M[s].items() if k != "base" and not k.startswith("_")}
        with open(f"history/{s}.json", "w") as f:
            json.dump(out, f, separators=(",", ":"))
        n += 1
    # The workflow only stages data.js, so stage the chart files here too.
    try:
        subprocess.run(["git", "add", "history"], check=False)
    except Exception as e:  # noqa
        print("git add history failed", e, file=sys.stderr)
    print(f"wrote {n} chart history files")


RATE = 0.04
NCDF = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, sig):
    d1 = (math.log(S / K) + (RATE + sig * sig / 2) * T) / (sig * math.sqrt(T)); d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-RATE * T) * NCDF(-d2) - S * NCDF(-d1), NCDF(d1) - 1


def implied_vol(px, S, K, T):
    lo, hi = 0.01, 5.0
    if px < bs_put(S, K, T, lo)[0] or px > bs_put(S, K, T, hi)[0]:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_put(S, K, T, mid)[0] > px: hi = mid
        else: lo = mid
    return (lo + hi) / 2


def put_options(s, last, asof):
    """~37-day puts near 20/30/40 delta. IV is solved from the option price so after-hours quotes stay sane."""
    try:
        t = yf.Ticker(s)
        exps = list(t.options or [])
        best = None
        for e in exps:
            dte = (dt.date.fromisoformat(e) - asof).days
            if dte < 7: continue
            score = abs(dte - 37) + (0 if 27 <= dte <= 65 else 100)  # stocks with only monthly options use the next monthly after ~4 weeks
            if best is None or score < best[0]: best = (score, e, dte)
        if not best: return s, None
        _, exp, dte = best
        puts = t.option_chain(exp).puts
        T = dte / 365
        prev_day = asof - dt.timedelta(days=4 if asof.weekday() == 0 else 1)
        rows = []
        for _, o in puts.iterrows():
            K = float(o["strike"])
            if not (0.6 * last <= K <= 1.05 * last): continue
            bid = float(o.get("bid") or 0); ask = float(o.get("ask") or 0); lp = float(o.get("lastPrice") or 0)
            ltd = o.get("lastTradeDate")
            try: fresh = pd.Timestamp(ltd).date() >= prev_day
            except Exception: fresh = False
            px = (bid + ask) / 2 if bid > 0 and ask > 0 and ask <= bid * 2 + 0.25 else (lp if fresh and lp > 0 else None)
            if not px or px < 0.05: continue
            iv = implied_vol(px, last, K, T)
            if not iv: continue
            delta = bs_put(last, K, T, iv)[1]
            rows.append({"k": K, "px": px, "bid": bid, "ask": ask, "iv": iv, "delta": delta,
                         "oi": int(o.get("openInterest") or 0) if not pd.isna(o.get("openInterest")) else 0})
        if len(rows) < 3: return s, None
        near = sorted(rows, key=lambda r: abs(r["k"] - last))[:2]
        atm = sum(r["iv"] for r in near) / len(near)
        picks = []
        for tgt in (0.20, 0.30, 0.40):
            r = min(rows, key=lambda r: abs(-r["delta"] - tgt))
            if abs(-r["delta"] - tgt) > 0.1: continue
            picks.append({"d": int(tgt * 100), "k": r2(r["k"]), "px": r2(r["px"]), "bid": r2(r["bid"]), "ask": r2(r["ask"]),
                          "iv": r2(r["iv"] * 100), "delta": r2(r["delta"]), "otm": r2((r["k"] / last - 1) * 100),
                          "ann": r2(r["px"] / r["k"] * 365 / dte * 100), "be": r2(r["k"] - r["px"]), "oi": r["oi"]})
        return s, {"exp": exp, "dte": dte, "atm": r2(atm * 100), "puts": picks}
    except Exception as e:  # noqa
        print("options fail", s, e, file=sys.stderr)
        return s, None


def load_json_js(path):
    try:
        t = open(path).read()
        return json.loads(t[t.index("=") + 1:].strip().rstrip(";"))
    except Exception:
        return None


def load_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    subprocess.run(["git", "add", path], check=False, stderr=subprocess.DEVNULL)


def track_record(frames, uni, M, D, asof):
    """Backfill: each past month's 50 worst (and 50 best) S&P members, and how they did afterwards."""
    spx = frames["^GSPC"]
    sd = [d.date() for d in spx.index]
    sc = {d: float(c) for d, c in zip(sd, spx["Close"])}
    ends = [sd[i] for i in range(len(sd) - 1) if sd[i].month != sd[i + 1].month]
    closes = {}
    for s in uni:
        if s in frames:
            closes[s] = {d.date(): float(c) for d, c in zip(frames[s].index, frames[s]["Close"])}
    pos = {d: i for i, d in enumerate(sd)}
    picks = load_json("picks.json", {})
    out = []
    for a, b in zip(ends, ends[1:]):
        rets = {s: c[b] / c[a] - 1 for s, c in closes.items() if a in c and b in c and c[a] > 0}
        if len(rets) < 400: continue
        mkey = b.strftime("%Y-%m")
        ranked = sorted(rets, key=rets.get)
        live = picks.get(mkey)
        losers = [x[0] for x in live["syms"]] if live else ranked[:50]
        winners = ranked[-50:]
        row = {"m": mkey, "end": b.isoformat(), "live": bool(live), "k": {}, "w": {}, "spx": {}}
        for h, key in ((21, "r1"), (63, "r3"), (126, "r6"), (None, "rd")):
            j = pos[b] + h if h else len(sd) - 1
            if j >= len(sd) or j == pos[b]: continue
            e = sd[j]
            sp = sc[e] / sc[b] - 1
            row["spx"][key] = r2(sp * 100)
            for grp, lst in (("k", losers), ("w", winners)):
                f = [closes[s][e] / closes[s][b] - 1 for s in lst if s in closes and b in closes[s] and e in closes[s]]
                if f:
                    row[grp][key] = r2(sum(f) / len(f) * 100)
                    row[grp]["b" + key[1:]] = round(sum(1 for x in f if x > sp) / len(f) * 100)
        if live:
            conf = {}
            for sym, cf in live["syms"]:
                if sym in closes and b in closes[sym]:
                    conf.setdefault(cf, []).append(closes[sym][sd[-1]] / closes[sym][b] - 1 if sd[-1] in closes[sym] else None)
            row["byConf"] = {k: [len(v), r2(sum(x for x in v if x is not None) / max(1, len([x for x in v if x is not None])) * 100)] for k, v in conf.items()}
        row["worst"] = [[s, r2(rets[s] * 100)] for s in ranked[:3]]
        out.append(row)
    # snapshot this month's actual list (last write of the month is the one that sticks)
    picks[asof.strftime("%Y-%m")] = {"asof": asof.isoformat(), "syms": [[r["s"], r["conf"]] for r in D]}
    write_json("picks.json", picks)
    return out[-18:]


def post_digest(text):
    tok, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not tok or not repo:
        print("digest (not posted, no token):\n" + text); return
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    api = f"https://api.github.com/repos/{repo}"
    title = "Falling Knives daily digest"
    try:
        iss = requests.get(f"{api}/issues", headers=h, params={"state": "open", "per_page": 100}, timeout=20).json()
        num = next((i["number"] for i in iss if i.get("title") == title), None)
        if num is None:
            num = requests.post(f"{api}/issues", headers=h, timeout=20, json={"title": title, "body":
                "The robot posts a comment here after every daily update. Watch this repo (or subscribe to this issue) to get it by email."}).json()["number"]
        r = requests.post(f"{api}/issues/{num}/comments", headers=h, json={"body": text}, timeout=20)
        print("digest posted", r.status_code)
    except Exception as e:  # noqa
        print("digest failed", e, file=sys.stderr)


def site_url():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo: return ""
    owner, name = repo.split("/")
    return f"https://{name}/" if name.lower().endswith(".github.io") else f"https://{owner.lower()}.github.io/{name}/"


def build_digest(meta, D, BC, G, V, prev_syms, asof):
    url = site_url()
    fmt = lambda v: ("+" if v > 0 else "") + f"{v:.1f}%"
    L = [f"## {asof.strftime('%A, %b %-d')} — S&P 500 {fmt(meta['spx_mtd'])} in {meta['monthName']}", ""]
    bk = (meta.get("basket") or {})
    if bk:
        bs = bk.get("syms", [])
        tag = f"**FINAL BASKET for {meta['monthName']}** (buy after the close, equal dollars each)" if bk.get("final") else "**This month's basket** (knives scoring 3+; locks at the month's last close)"
        L.append(tag)
        L.append(("- " + ", ".join(f"**{a}** ({c}/4, {fmt(bk.get('drop', {}).get(a, 0))})" for a, _, c in bs)) if bs else "- Empty so far: no knife scores 3+ (the rule sits out)")
        L.append("")
    new = [r for r in D if r["s"] not in prev_syms] if prev_syms else []
    if new:
        L.append("**New falling knives today**")
        L += [f"- **{r['s']}** {r['name']} — {fmt(r['mtd'])} this month, No. {r['rank']}" for r in new[:10]]
        L.append("")
    reb = [r for r in D + BC if len(r.get("sig") or []) >= 2]
    seen = set(); reb = [r for r in reb if not (r["s"] in seen or seen.add(r["s"]))]
    if reb:
        L.append("**Knives showing 2+ bottoming signs**")
        L += [f"- **{r['s']}** {fmt(r['mtd'])} — " + ", ".join(r["sig"]) for r in reb[:10]]
        L.append("")
    soon = []
    for r in D + BC + G + V:
        if r.get("ed"):
            dd = (dt.date.fromisoformat(r["ed"]) - asof).days
            if 0 <= dd <= 7 and r["s"] not in [x[0] for x in soon]: soon.append((r["s"], r["ed"], dd))
    if soon:
        L.append("**Earnings in the next 7 days**")
        L += [f"- **{s}** {dt.date.fromisoformat(e).strftime('%a %b %-d')}" + (" (tomorrow)" if d == 1 else " (today)" if d == 0 else "") for s, e, d in sorted(soon, key=lambda x: x[2])[:12]]
        L.append("")
    pat = [r for r in D + BC if r.get("pat") and r["pat"]["score"] >= 3]
    seen = set(); pat = [r for r in sorted(pat, key=lambda r: -r["pat"]["score"]) if not (r["s"] in seen or seen.add(r["s"]))]
    if pat:
        L.append("**Knives matching the perfect-pick pattern (3+ of 4)**")
        names = {"t": "tech", "c": "group sell-off", "n": "not company-specific", "d": "30%+ off high"}
        L += [f"- **{r['s']}** {fmt(r['mtd'])} — {r['pat']['score']}/4 (" + ", ".join(v for k, v in names.items() if r['pat'][k]) + ")" for r in pat[:10]]
        L.append("")
    rich = [r for r in D + BC if r.get("pAnn")]
    seen = set(); rich = [r for r in sorted(rich, key=lambda r: -r["pAnn"]) if not (r["s"] in seen or seen.add(r["s"]))]
    if rich:
        L.append("**Richest 30-delta puts on the falling lists (~37 days)**")
        for r in rich[:5]:
            p = next(x for x in r["opt"]["puts"] if x["d"] == 30)
            L.append(f"- **{r['s']}** ${p['k']:g} put ({abs(p['otm']):.0f}% below) · ${p['px']:.2f} · {p['ann']:.0f}% annualized · IV {r['opt']['atm']:.0f}%" + (f" · earnings {r['ed']}" if r.get("ed") and asof.isoformat() <= r["ed"] <= r["opt"]["exp"] else ""))
        L.append("")
    if url: L.append(f"[Open the site]({url})")
    L.append("\n<sub>Not investment advice. Options prices are from the close and can be stale; check your broker.</sub>")
    return "\n".join(L)


def _easter(y):
    a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31; da = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(y, mo, da)


def nyse_holidays(y):
    def obs(d):  # weekend -> observed weekday
        return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d
    def nth(month, weekday, n):
        d = dt.date(y, month, 1); d += dt.timedelta(days=(weekday - d.weekday()) % 7); return d + dt.timedelta(weeks=n - 1)
    def last(month, weekday):
        d = dt.date(y + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1); return d - dt.timedelta(days=(d.weekday() - weekday) % 7)
    hs = {obs(dt.date(y, 1, 1)), nth(1, 0, 3), nth(2, 0, 3), _easter(y) - dt.timedelta(days=2), last(5, 0),
          obs(dt.date(y, 6, 19)), obs(dt.date(y, 7, 4)), nth(9, 0, 1), nth(11, 3, 4), obs(dt.date(y, 12, 25))}
    return hs


def is_trading_day(d):
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def is_last_trading_day(d):
    n = d + dt.timedelta(days=1)
    while not is_trading_day(n):
        n += dt.timedelta(days=1)
    return n.month != d.month


def _lock_state(last_date):
    try:
        from zoneinfo import ZoneInfo
        now_ny = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # noqa
        now_ny = dt.datetime.utcnow() - dt.timedelta(hours=4)
    after_close = last_date < now_ny.date() or now_ny.hour * 60 + now_ny.minute >= 16 * 60 + 5
    return is_last_trading_day(last_date) and after_close


def update_pick_log(D, last_date):
    """Monthly basket = every knife with pick-pattern score 3+ (equal dollars each).
    The top of the basket (highest score, ties -> biggest drop) is also logged as the single pick.
    Both lock at the close on the month's last trading day."""
    log = load_json("picklog.json", {})
    blog = load_json("basketlog.json", {})
    key = last_date.strftime("%Y-%m")
    for L in (log, blog):
        for k, v in L.items():
            if k < key and not v.get("final"):
                v["final"] = True
    cands = sorted([r for r in D if r.get("pat")], key=lambda r: (-r["pat"]["score"], r["mtd"]))
    final = _lock_state(last_date)
    def keep(prev):
        return prev and prev.get("final") and prev.get("src") == "live" and prev.get("asof", "") >= last_date.isoformat() and not final
    if cands and not keep(log.get(key)):
        lead = cands[0]
        log[key] = {"s": lead["s"], "name": lead["name"], "score": lead["pat"]["score"], "drop": lead["mtd"], "px": lead["last"],
                    "chk": {k: lead["pat"][k] for k in "tcnd"}, "src": "live", "yf": lead["s"], "asof": last_date.isoformat(),
                    "final": final, "next": [[r["s"], r["pat"]["score"], r["mtd"]] for r in cands[1:3]]}
    if not keep(blog.get(key)):
        members = [r for r in cands if r["pat"]["score"] >= 3]
        blog[key] = {"syms": [[r["s"], r["s"], r["pat"]["score"]] for r in members], "src": "live",
                     "asof": last_date.isoformat(), "final": final,
                     "px": {r["s"]: r["last"] for r in members}, "drop": {r["s"]: r["mtd"] for r in members}}
    write_json("picklog.json", log)
    write_json("basketlog.json", blog)
    return log, blog, (key if cands else None)


def _month_end(k):
    y, m = map(int, k.split("-"))
    return pd.Timestamp(dt.date(y + (m == 12), m % 12 + 1, 1)) - pd.Timedelta(days=1)


def _irr(flows, end_value, now):
    """Annual return on money added over time (flows: list of (Timestamp, amount))."""
    if not flows or end_value <= 0: return None
    lo, hi = -0.95, 5.0
    f = lambda r: sum(c * (1 + r) ** ((now - d).days / 365.25) for d, c in flows) - end_value
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0: hi = mid
        else: lo = mid
    return r2((lo + hi) / 2 * 100)


def pick_performance(log, blog, spy_frame):
    """Returns for the single-pick log (1 share each) and the basket log ($100 per stock), plus SPY over the same spans."""
    syms = sorted({v.get("yf") or v["s"] for v in log.values()} | {x[1] for v in blog.values() for x in v.get("syms", [])} | {"SPY"})
    closes = {}
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        try:
            raw = yf.download(chunk, period="11y", interval="1d", auto_adjust=False, group_by="ticker", threads=True, progress=False)
        except Exception as e:  # noqa
            print("pick download failed", e, file=sys.stderr); continue
        for sym in chunk:
            try:
                sub = raw[sym] if len(chunk) > 1 else raw
                ser = sub["Close"].dropna()
                if len(ser): closes[sym] = ser
            except Exception:
                pass
    spy = closes.get("SPY")
    if spy is None and spy_frame is not None:
        spy = spy_frame["Close"].dropna()
    def ret_since(ser, d):
        if ser is None or not len(ser): return None
        b = ser[ser.index <= d]
        return float(ser.iloc[-1] / b.iloc[-1] - 1) if len(b) else None
    now = pd.Timestamp(spy.index[-1]) if spy is not None else pd.Timestamp.today()
    # single pick
    out, cost, val, spyv, up, beat, fl = [], 0.0, 0.0, 0.0, 0, 0, []
    for k in sorted(log):
        v = dict(log[k]); d = pd.Timestamp(v["asof"]) if v.get("asof") else _month_end(k)
        ret, sret = ret_since(closes.get(v.get("yf") or v["s"]), d), ret_since(spy, d)
        v.update(m=k, ret=r2(ret * 100) if ret is not None else None, spyRet=r2(sret * 100) if sret is not None else None,
                 now=r2(v["px"] * (1 + ret)) if ret is not None else None)
        if ret is not None and v.get("final"):
            cost += v["px"]; val += v["px"] * (1 + ret); spyv += v["px"] * (1 + (sret or 0)); fl.append((d, v["px"]))
            up += ret > 0; beat += sret is not None and ret > sret
        out.append(v)
    n = sum(1 for v in out if v.get("final") and v.get("ret") is not None)
    tot = {"n": n, "cost": r2(cost), "val": r2(val), "spy": r2(spyv), "up": up, "beat": beat,
           "irr": _irr(fl, val, now), "irrSpy": _irr(fl, spyv, now),
           "live": sum(1 for v in out if v.get("src") == "live" and v.get("final"))}
    # basket ($100 per stock)
    bout, bc, bv, bs, bfl, buys, months, bup, bbeat, missing = [], 0.0, 0.0, 0.0, [], 0, 0, 0, 0, 0
    for k in sorted(blog):
        v = blog[k]; d = pd.Timestamp(v["asof"]) if v.get("asof") else _month_end(k)
        sret = ret_since(spy, d)
        rets = []
        for s_, yfs, sc in v.get("syms", []):
            r = ret_since(closes.get(yfs), d)
            if r is None: missing += 1
            else: rets.append([s_, sc, r2(r * 100)])
        row = {"m": k, "n": len(v.get("syms", [])), "src": v.get("src"), "final": v.get("final"), "asof": v.get("asof"),
               "syms": [x[0] for x in v.get("syms", [])], "spyRet": r2(sret * 100) if sret is not None else None}
        if rets:
            avg = sum(x[2] for x in rets) / len(rets)
            row.update(ret=r2(avg), inv=100 * len(rets), now=r2(100 * len(rets) * (1 + avg / 100)),
                       best=max(rets, key=lambda x: x[2]), worst=min(rets, key=lambda x: x[2]))
            if v.get("final"):
                months += 1; buys += len(rets); bc += 100 * len(rets); bv += row["now"]; bs += 100 * len(rets) * (1 + (sret or 0))
                bfl.append((d, 100 * len(rets))); bup += avg > 0; bbeat += sret is not None and avg > sret * 100
        bout.append(row)
    btot = {"months": months, "buys": buys, "cost": r2(bc), "val": r2(bv), "spy": r2(bs), "irr": _irr(bfl, bv, now),
            "irrSpy": _irr(bfl, bs, now), "up": bup, "beat": bbeat, "missing": missing,
            "sitout": sum(1 for v in blog.values() if v.get("final") and not v.get("syms")),
            "live": sum(1 for v in blog.values() if v.get("src") == "live" and v.get("final"))}
    return {"log": out[::-1], "tot": tot, "blog": bout[::-1], "btot": btot}


def post_final_issue(bentry, pentry, month_name):
    """One email-triggering issue per month when the basket locks."""
    tok, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    syms = bentry.get("syms", [])
    if syms:
        title = f"FINAL basket for {month_name}: {len(syms)} stock{'s' if len(syms) != 1 else ''} (top: {pentry['s'] if pentry else syms[0][0]})"
        lines = [f"- **{s_}** — score {sc}/4, {bentry.get('drop', {}).get(s_, 0):+.1f}% this month, ${bentry.get('px', {}).get(s_, 0):.2f}" for s_, _, sc in syms]
        body = (f"Every falling knife scoring 3+ on the pick pattern at the {month_name} close. Split your money equally (for example $100 each = ${100 * len(syms):,}).\n\n"
                + "\n".join(lines) + "\n\n")
    else:
        title = f"FINAL basket for {month_name}: empty — no knives scored 3+ (sit out)"
        body = "No falling knife scored 3 or more on the pick pattern this month, so the rule sits out.\n\n"
    body += (f"[Open the site]({site_url()})\n\n" if site_url() else "") + "<sub>Rule-based basket from a backtest, not investment advice. Check live prices and news before buying.</sub>"
    if not tok or not repo:
        print("final basket (not posted):", title); return
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    try:
        api = f"https://api.github.com/repos/{repo}"
        existing = requests.get(f"{api}/issues", headers=h, params={"state": "all", "per_page": 50}, timeout=20).json()
        if any(i.get("title", "").startswith(f"FINAL basket for {month_name}:") or i.get("title", "").startswith(f"FINAL pick for {month_name}:") for i in existing):
            return
        r = requests.post(f"{api}/issues", headers=h, json={"title": title, "body": body}, timeout=20)
        print("final basket issue", r.status_code)
    except Exception as e:  # noqa
        print("final basket issue failed", e, file=sys.stderr)


def prev_month_end(d):
    e = d.replace(day=1) - dt.timedelta(days=1)
    while not is_trading_day(e):
        e -= dt.timedelta(days=1)
    return e


def finalize_missed_month(frames, uni, last_date, reasons):
    """If the run on last month's final trading day was missed (failed or stale data), rebuild
    last month's basket from the true month-end close so it locks on the right prices."""
    pe = prev_month_end(last_date)
    key = pe.strftime("%Y-%m")
    blog = load_json("basketlog.json", {})
    cur = blog.get(key)
    if cur and cur.get("src") != "live":
        return
    if cur and cur.get("final") and cur.get("asof") == pe.isoformat():
        return
    fr = {s: _upto(f, pe) for s, f in frames.items()}
    if "^GSPC" not in fr or not len(fr["^GSPC"]) or fr["^GSPC"].index[-1].date() != pe:
        return
    ms = pe.replace(day=1)
    Mp = {}
    for s in uni:
        f = fr.get(s)
        if f is not None and len(f) > 60 and f.index[-1].date() == pe:
            try:
                Mp[s] = metrics(s, f, ms)
            except Exception:
                pass
    if len(Mp) < 450:
        return
    spx = metrics("^GSPC", fr["^GSPC"], ms)
    secmed = {}
    for sec in {u["sector"] for u in uni.values()}:
        v = sorted(Mp[x]["mtd"] for x in Mp if uni[x]["sector"] == sec)
        if v: secmed[sec] = v[len(v) // 2]
    rows = []
    for r in sorted(Mp.values(), key=lambda r: r["mtd"])[:TOP_N]:
        s = r["s"]
        rows.append(dict(r, name=uni[s]["name"], pat=pick_pattern(r, uni[s]["sector"], uni[s].get("sub"), secmed, spx["mtd"])))
    print(f"catch-up: rebuilding {key} basket from the {pe} close")
    update_pick_log(rows, pe)
    picks = load_json("picks.json", {})
    conf = lambda s: reasons[s][1] if isinstance(reasons.get(s), list) and len(reasons[s]) == 3 else "Pending"
    picks[key] = {"asof": pe.isoformat(), "syms": [[r["s"], conf(r["s"])] for r in rows]}
    write_json("picks.json", picks)


def auto_stories(D, month_name):
    themes = {}
    for r in D:
        if r["theme"] == "Reason pending": continue
        themes.setdefault(r["theme"], []).append(r)
    top = sorted(themes.items(), key=lambda kv: -len(kv[1]))[:3]
    out = []
    for th, rows in top:
        rows = sorted(rows, key=lambda r: r["mtd"])
        w = rows[0]
        out.append({"k": f"{len(rows)} of the 50", "h": f"{th}.",
                    "p": f"The biggest single theme behind this {month_name} list. Hardest hit: {w['name']} ({w['s']}), down {abs(w['mtd']):.1f}% this month.",
                    "t": " · ".join(r["s"] for r in rows[:8])})
    return out


def main():
    uni = load_universe()
    syms = sorted(uni)
    frames = download(syms + ["^GSPC", "SPY"])
    frames, last_date = align_frames(frames, syms + ["^GSPC", "SPY"])
    month_start = last_date.replace(day=1)
    M = {}
    for s in syms:
        if s in frames and frames[s].index[-1].date() == last_date:
            try:
                M[s] = metrics(s, frames[s], month_start)
            except Exception as e:
                print("metrics fail", s, e, file=sys.stderr)
    print(f"computed {len(M)} of {len(syms)} as of {last_date}")
    if len(M) < 450:
        sys.exit("Too few symbols computed; not publishing.")
    spx = metrics("^GSPC", frames["^GSPC"], month_start)
    spy = metrics("SPY", frames["SPY"], month_start) if "SPY" in frames else None

    sdr = spx["_dr"]
    for r in M.values():
        pairs = [(v, sdr[d]) for d, v in r["_dr"].items() if d in sdr]
        if len(pairs) > 60:
            mx = sum(p[1] for p in pairs) / len(pairs); my = sum(p[0] for p in pairs) / len(pairs)
            cov = sum((p[0] - my) * (p[1] - mx) for p in pairs); var = sum((p[1] - mx) ** 2 for p in pairs)
            r["beta"] = r2(cov / var) if var else None
        else:
            r["beta"] = None

    secmed = {}
    for sec in {u["sector"] for u in uni.values()}:
        v = sorted(M[x]["mtd"] for x in M if uni[x]["sector"] == sec)
        if v: secmed[sec] = v[len(v) // 2]

    caps = market_caps(syms)
    ranked = sorted(M.values(), key=lambda r: r["mtd"])
    top = ranked[:TOP_N]
    gain = sorted(M.values(), key=lambda r: -r["mtd"])[:GAIN_N]
    vol = sorted([r for r in M.values() if r["vol30"]], key=lambda r: -r["vol30"])[:VOL_N]
    big = sorted([s for s in syms if caps.get(s) and s not in SKIP_BIG and s in M],
                 key=lambda s: -caps[s])[:BIG_N]
    bc = sorted([M[s] for s in big if M[s]["mtd"] < 0], key=lambda r: r["mtd"])
    vbc = sorted([M[s] for s in big if M[s]["vol30"]], key=lambda r: -r["vol30"])
    card_syms = sorted({r["s"] for r in top} | {r["s"] for r in bc} | {r["s"] for r in gain} | {r["s"] for r in vol})
    with ThreadPoolExecutor(6) as ex:
        det = dict(ex.map(details, card_syms))
    opt_syms = sorted({r["s"] for r in top} | {r["s"] for r in bc} | set(big) | {r["s"] for r in vol})
    with ThreadPoolExecutor(4) as ex:
        opts = dict(ex.map(lambda s: put_options(s, M[s]["last"], last_date), opt_syms))
    ivh = load_json("ivhist.json", {})
    for s, o in opts.items():
        if not o: continue
        h = [x for x in ivh.get(s, []) if x[0] != last_date.isoformat()] + [[last_date.isoformat(), o["atm"]]]
        ivh[s] = h[-260:]
        vals = [x[1] for x in ivh[s]]
        o["ivn"] = len(vals)
        o["ivr"] = round((o["atm"] - min(vals)) / (max(vals) - min(vals)) * 100) if len(vals) >= 20 and max(vals) > min(vals) else None
    write_json("ivhist.json", ivh)
    print(f"options: {sum(1 for o in opts.values() if o)} of {len(opt_syms)}")

    try:
        reasons = json.load(open("reasons.json"))
    except FileNotFoundError:
        reasons = {}

    def card(r, rank, kind="fall"):
        s = r["s"]; o = {k: v for k, v in r.items() if k not in ("base", "wkends") and not k.startswith("_")}
        o.update(det.get(s, {}))
        o["name"] = uni[s]["name"]; o["sector"] = uni[s]["sector"]; o["rank"] = rank
        o["sub"] = uni[s].get("sub") or None
        o["pat"] = pick_pattern(r, uni[s]["sector"], uni[s].get("sub"), secmed, spx["mtd"])
        o["mc"] = r2(caps[s] / 1e9) if caps.get(s) else None
        op = opts.get(s)
        if op:
            o["opt"] = op
            p30 = next((x for x in op["puts"] if x["d"] == 30), None)
            o["pAnn"] = p30["ann"] if p30 else None
            o["ivhv"] = r2(op["atm"] / r["vol30"]) if r.get("vol30") else None
            o["ivr"] = op.get("ivr")
        rs = reasons.get(s)
        if kind == "gain" or (kind == "vol" and not (rs and r["mtd"] < 0)):
            o["theme"], o["conf"], o["why"] = "In the news", "News", ""
        elif rs and isinstance(rs, list) and len(rs) == 3:
            o["theme"], o["conf"], o["why"] = rs
            o.pop("news", None)
        else:
            o["theme"], o["conf"], o["why"] = "Reason pending", "Pending", ""
        return o

    D = [card(r, i + 1) for i, r in enumerate(top)]
    capRank = {s: i + 1 for i, s in enumerate(big)}
    BC = [card(r, i + 1) for i, r in enumerate(bc)]
    for o in BC: o["capRank"] = capRank[o["s"]]
    G = [card(r, i + 1, "gain") for i, r in enumerate(gain)]
    V = [card(r, i + 1, "vol") for i, r in enumerate(vol)]
    VB = [card(r, i + 1, "vol") for i, r in enumerate(vbc)]
    for o in VB: o["capRank"] = capRank[o["s"]]
    BIG = [[s, round(caps[s] / 1e9), M[s]["mtd"]] for s in big]
    BASE = {s: M[s]["base"] for s in set(card_syms) | set(big)}
    if spy: BASE["SPY"] = spy["base"]

    month_name = calendar.month_name[last_date.month]
    stories_cfg = reasons.get("_stories", {})
    stories = stories_cfg.get("tiles") if stories_cfg.get("month") == last_date.strftime("%Y-%m") else None
    all_mtd = sorted(r["mtd"] for r in M.values())
    meta = {
        "asof": last_date.isoformat(), "monthName": month_name, "year": last_date.year,
        "secMed": {k: r2(v) for k, v in secmed.items()},
        "spx_mtd": spx["mtd"], "ndown": sum(1 for x in all_mtd if x < 0), "n": len(M),
        "median": all_mtd[len(all_mtd) // 2], "nup": sum(1 for x in all_mtd if x > 0),
        "spx_vol30": spx["vol30"],
        "medVol": sorted(r["vol30"] for r in M.values() if r["vol30"])[len(M) // 2 - 1],
        "wkdates": ["Wk of " + x for x in top[0]["wkends"][:-1]] + ["This week (so far)"],
        "stories": stories or auto_stories(D, month_name),
        "storySub": stories_cfg.get("sub") if stories else "The biggest themes behind this list.",
        "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    write_history(sorted(set(card_syms) | set(big)), frames, uni, M)
    prev = load_json_js("data.js")
    prev_syms = {r["s"] for r in prev.get("D", [])} if prev else set()
    TR = track_record(frames, uni, M, D, last_date)
    # compact file with every S&P member, for the watchlist and search
    ALLS = {}
    for s, r in M.items():
        o = {k: v for k, v in r.items() if k not in ("base", "wkends") and not k.startswith("_")}
        o["name"] = uni[s]["name"]; o["sector"] = uni[s]["sector"]; o["mc"] = r2(caps[s] / 1e9) if caps.get(s) else None
        o["pat"] = pick_pattern(r, uni[s]["sector"], uni[s].get("sub"), secmed, spx["mtd"])
        f = frames[s]; wk = f["Close"].resample("W-FRI").last().dropna().tail(53)
        o["wc"] = [[d.strftime("%Y-%m-%d"), r2(v)] for d, v in wk.items()]
        ALLS[s] = o
    write_json("all.json", {"asof": last_date.isoformat(), "S": ALLS, "BASE": {s: M[s]["base"] for s in M}})
    try:
        finalize_missed_month(frames, uni, last_date, reasons)
    except Exception as e:  # noqa
        print("month-end catch-up failed", e, file=sys.stderr)
    plog, blog, pkey = update_pick_log(D, last_date)
    PK = pick_performance(plog, blog, frames.get("SPY"))
    PK["cur"] = dict(plog[pkey], m=pkey) if pkey else None
    bkey = last_date.strftime("%Y-%m")
    PK["bcur"] = dict(blog.get(bkey, {}), m=bkey)
    meta["pick"] = PK["cur"]; meta["basket"] = PK["bcur"]
    finals = [k for k, v in blog.items() if v.get("final") and v.get("src") == "live"]
    if finals:
        fk = max(finals); fy, fm = map(int, fk.split("-"))
        post_final_issue(blog[fk], plog.get(fk), f"{calendar.month_name[fm]} {fy}")
    payload = {"D": D, "BC": BC, "BIG": BIG, "M": meta, "BASE": BASE, "G": G, "V": V, "VB": VB, "TR": TR, "PK": PK}
    with open("data.js", "w") as f:
        f.write("window.FK=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    if prev and prev.get("M", {}).get("asof") == meta["asof"]:
        print("digest skipped: already sent for", meta["asof"])
    else:
        try:
            post_digest(build_digest(meta, D, BC, G, V, prev_syms, last_date))
        except Exception as e:  # noqa
            print("digest build failed", e, file=sys.stderr)
    pend = [o["s"] for o in D + BC if o["conf"] == "Pending"]
    print(f"wrote data.js: {len(D)} knives, {len(BC)} falling blue chips, {len(G)} winners, {len(V)} volatile, {len(VB)} blue chips by volatility; pending reasons: {pend}")


if __name__ == "__main__":
    main()

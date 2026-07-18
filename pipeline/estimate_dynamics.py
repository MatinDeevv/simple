"""
estimate_dynamics.py -- causal per-pair estimation of single-particle dynamics
parameters for the FX simulator (OQ-1..OQ-5a; spec: docs/dynamics.md).

Structural equation (schema section 4, coupling term excluded -- owned elsewhere):

    m_i dv_i/dt = -k_i (x_i - x_eq,i) - c_i v_i + F_i

Estimated here in specific (per-unit-mass) form with strictly causal,
beginning-of-step regressors:

    a_i(t) = -kappa_i(t) d_i(t-1) - gamma_i(t) v_i(t-1) + eps_i(t)

    d      = x - x_eq                       [nats]
    kappa  = k/m                            [s^-2]
    gamma  = c/m                            [s^-1]
    eps    = F/m  (specific forcing)        [nats s^-2]

Definitions (all causal, lookahead rule schema section 5):
    x_eq,i(t)   : wall-clock EWMA of x, half-life HL_EQ_S            (OQ-2)
    m_i(t)      : clip( (SIGMA_STAR / sigma_hat_i(t))^2, M_CLIP )    (OQ-1)
                  sigma_hat^2 = wall-clock EWMA of valid 1-min squared
                  log returns, half-life HL_VOL_S (equipartition mass)
    kappa,gamma : rolling no-intercept OLS over trailing OLS_WINDOW bars,
                  gap-crossing observations excluded                 (OQ-3, OQ-4)
    k, c        : m*kappa, m*gamma                                   (OQ-3, OQ-4)
    F_i(t)      : m_i(t) * eps_i(t)  (estimation-mode residual)      (OQ-5a)
    sim-mode F  : m * sigma_eps(t) * sqrt(s(how)) * eta,
                  eta ~ iid standardized Student-t(max(nu_fit, 2.1)) (OQ-5a)

Gap handling at estimation time: any observation whose backward differences
cross a bar gap (dt != 60 s at t or t-1) is excluded from the OLS, from the
volatility EWMA and from residual fitting.  No velocity is fabricated across
gaps.  Integrator gap policy is OQ-8 and is NOT decided here.

Run:    python pipeline\\estimate_dynamics.py            (EURUSD USDJPY USDCNH)
        python pipeline\\estimate_dynamics.py --all      (all 10 pairs)
        python pipeline\\estimate_dynamics.py --pairs EURUSD --sensitivity

Outputs:
    data_derived/dynamics_params_<PAIR>.parquet   daily-sampled parameter stream
    data_derived/dynamics_summary.json            parameter ranges + diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import canonical_pair_order

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_canonical"
OUT_DIR = ROOT / "data_derived"

ESTIMATOR_VERSION = "dynamics-est-1.2.0"

# ----- frozen conventions (a-priori constants, not fitted from data) ---------
DT_NOM = 60.0                    # nominal bar interval [s]
SIGMA_STAR = 1e-4                # mass anchor: reference 1-min return scale [nats]
M_CLIP = (0.1, 10.0)             # mass clip band (dimensionless)
HL_EQ_S = 24 * 3600.0            # x_eq EWMA half-life [s], wall clock       (OQ-2)
HL_VOL_S = 5 * 24 * 3600.0       # sigma_hat^2 EWMA half-life [s], wall clock (OQ-1)
HL_EPS_S = 5 * 24 * 3600.0       # sigma_eps^2 EWMA half-life [s], wall clock (OQ-5a)
OLS_WINDOW = 43_200              # trailing bars in rolling OLS (~30 trading days)
MIN_OBS = 20_000                 # min valid obs in window before params are emitted
SEASON_BINS = 168                # hour-of-week bins
SEASON_MIN_COUNT = 100           # min prior obs in a bin before s != 1
SEASON_CLIP = (0.1, 10.0)
NU_FIT_MAX_POINTS = 300_000      # subsample cap for Student-t MLE
NU_SIM_MIN = 2.1                 # ensure simulated Student-t has finite variance
ACF_LAGS = (1, 2, 3, 5, 10, 30)
PCTL = (5, 25, 50, 75, 95)

PAIRS_ALL = list(canonical_pair_order(ROOT))
PAIRS_DEFAULT = ["EURUSD", "USDJPY", "USDCNH"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def shift1(arr: np.ndarray) -> np.ndarray:
    """Backward shift by one row (row 0 -> NaN). Causal."""
    out = np.empty_like(arr)
    out[0] = np.nan
    out[1:] = arr[:-1]
    return out


def ewm_wall(values: np.ndarray, times: pd.DatetimeIndex, halflife_s: float) -> np.ndarray:
    """Normalized EWMA with wall-clock half-life: weights 2^{-(t-tau)/HL}.
    Causal by construction (weights only on tau <= t)."""
    s = pd.Series(values)
    return (
        s.ewm(halflife=pd.Timedelta(seconds=halflife_s), times=times)
        .mean()
        .to_numpy()
    )


def windowed_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum over the last `window` rows (expanding until full).
    arr must contain no NaN (mask upstream with np.where)."""
    cs = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    idx_hi = np.arange(1, arr.size + 1)
    idx_lo = np.maximum(0, idx_hi - window)
    return cs[idx_hi] - cs[idx_lo]


def rolling_ols_2reg(y, r1, r2, mask, window, min_obs):
    """Causal rolling no-intercept OLS  y ~ b1*r1 + b2*r2  over trailing
    `window` rows, restricted to rows where mask is True.
    Returns (b1, b2, b2_only, n_valid): b2_only is the damping-only 1-regressor
    slope y ~ b2*r2 on the same windows (spring-increment diagnostic)."""
    mz = mask.astype(np.float64)
    z = lambda a: np.where(mask, a, 0.0)
    y0, r10, r20 = z(y), z(r1), z(r2)

    S11 = windowed_sum(r10 * r10, window)
    S22 = windowed_sum(r20 * r20, window)
    S12 = windowed_sum(r10 * r20, window)
    S1y = windowed_sum(r10 * y0, window)
    S2y = windowed_sum(r20 * y0, window)
    nv = windowed_sum(mz, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        det = S11 * S22 - S12 * S12
        ok = (nv >= min_obs) & (det > 0) & (S11 > 0) & (S22 > 0)
        b1 = np.where(ok, (S1y * S22 - S2y * S12) / det, np.nan)
        b2 = np.where(ok, (S2y * S11 - S1y * S12) / det, np.nan)
        b2_only = np.where(ok, S2y / S22, np.nan)
    return b1, b2, b2_only, nv


def fit_student_t(z: np.ndarray) -> dict:
    """MLE of Student-t dof on standardized residuals z (loc fixed 0).
    scipy if available, else grid MLE over (nu, scale)."""
    sub = z[:: max(1, z.size // NU_FIT_MAX_POINTS)]
    try:
        from scipy import stats

        nu, _loc, sc = stats.t.fit(sub, floc=0.0)
        return {"nu": float(nu), "scale": float(sc),
                "implied_var": float(sc * sc * nu / (nu - 2.0)) if nu > 2 else None,
                "n_points": int(sub.size), "method": "scipy-mle"}
    except ImportError:
        pass
    nus = np.concatenate([np.arange(2.2, 12.01, 0.2), np.arange(13.0, 31.0)])
    scales = np.exp(np.linspace(math.log(0.4), math.log(2.5), 25))
    best = (-np.inf, np.nan, np.nan)
    for nu in nus:
        cst = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
               - 0.5 * math.log(nu * math.pi))
        for sc in scales:
            u = sub / sc
            ll = sub.size * (cst - math.log(sc)) \
                 - (nu + 1) / 2 * np.log1p(u * u / nu).sum()
            if ll > best[0]:
                best = (ll, nu, sc)
        _, nu_b, sc_b = best
    return {"nu": float(nu_b), "scale": float(sc_b),
            "implied_var": float(sc_b * sc_b * nu_b / (nu_b - 2.0)) if nu_b > 2 else None,
            "n_points": int(sub.size), "method": "grid-mle"}


def acf_gap_aware(e: np.ndarray, epoch: np.ndarray, lag: int) -> float | None:
    """Autocorrelation of e at `lag` bars using only pairs exactly 60*lag s
    apart with both values finite (no correlation across gaps)."""
    n = e.size
    i = np.arange(lag, n)
    j = i - lag
    fin = np.isfinite(e)
    m = fin[i] & fin[j] & ((epoch[i] - epoch[j]) == 60 * lag)
    if m.sum() < 1000:
        return None
    a, b = e[i[m]], e[j[m]]
    return float(np.corrcoef(a, b)[0, 1])


def pctls(arr: np.ndarray) -> dict:
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(fin, p)) for p in PCTL}


# --------------------------------------------------------------------------- #
# per-pair estimation
# --------------------------------------------------------------------------- #

def process_pair(pair: str, sensitivity: bool = False) -> dict:
    t0 = time.time()
    path = DATA_DIR / f"{pair}.parquet"
    df = pd.read_parquet(path, columns=["timestamp", "close"])
    ts = pd.DatetimeIndex(df["timestamp"])
    epoch = ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    x = np.log(df["close"].to_numpy(dtype=np.float64))
    n = x.size
    del df

    # -- bookkeeping: dt, gap flags, validity masks (schema section 4.2) ------
    dt = np.empty(n, dtype=np.int64)
    dt[0] = -1                                # first bar: no predecessor
    dt[1:] = epoch[1:] - epoch[:-1]
    ret_valid = dt == 60                      # v(t) computable without gap
    obs_valid = ret_valid & shift1(ret_valid.astype(np.float64)).astype(bool)
    obs_valid[:2] = False                     # a(t) and v(t-1) both gap-free

    # -- kinematics: backward differences only (schema section 5) ------------
    xl1 = shift1(x)
    r = x - xl1                               # 1-min log return [nats]
    v = np.where(ret_valid, r / DT_NOM, np.nan)          # [nats/s]
    vlag = shift1(v)
    a = np.where(obs_valid, (v - vlag) / DT_NOM, np.nan)  # [nats/s^2]

    # -- OQ-2: equilibrium = wall-clock EWMA of x, half-life 24 h -------------
    x_eq = ewm_wall(x, ts, HL_EQ_S)
    d = x - x_eq                              # displacement [nats]
    dlag = shift1(d)

    # -- OQ-1: equipartition mass from causal EWMA realized variance ----------
    vidx = np.flatnonzero(ret_valid)
    sig2 = np.full(n, np.nan)
    sig2[vidx] = ewm_wall(r[vidx] ** 2, ts[vidx], HL_VOL_S)
    sig2 = pd.Series(sig2).ffill().to_numpy()            # hold across gaps
    sigma_hat = np.sqrt(sig2)                            # [nats per 1-min bar]
    with np.errstate(divide="ignore", invalid="ignore"):
        m_raw = SIGMA_STAR ** 2 / sig2
    m = np.clip(m_raw, M_CLIP[0], M_CLIP[1])
    frac_m_clipped = float(np.mean((m_raw < M_CLIP[0]) | (m_raw > M_CLIP[1])))

    # -- OQ-3/OQ-4: rolling causal OLS  a(t) ~ -kappa d(t-1) - gamma v(t-1) ---
    b1, b2, b2_only, nv = rolling_ols_2reg(a, dlag, vlag, obs_valid,
                                           OLS_WINDOW, MIN_OBS)
    kappa = -b1                               # k/m [s^-2]
    gamma = -b2                               # c/m [s^-1]
    gamma_only = -b2_only
    k = m * kappa                             # [s^-2]
    c = m * gamma                             # [s^-1]

    # -- OQ-5a estimation-mode residual (specific forcing) --------------------
    eps = np.where(obs_valid & np.isfinite(kappa),
                   a + kappa * dlag + gamma * vlag, np.nan)
    eps_damp_only = np.where(obs_valid & np.isfinite(gamma_only),
                             a + gamma_only * vlag, np.nan)
    F = m * eps                               # [nats/s^2]

    fin_e = np.isfinite(eps)
    sse_full = float(np.nansum(eps[fin_e] ** 2))
    sse_damp = float(np.nansum(eps_damp_only[fin_e] ** 2))
    sst = float(np.nansum(a[fin_e] ** 2))
    r2_full = 1.0 - sse_full / sst if sst > 0 else None
    r2_damp = 1.0 - sse_damp / sst if sst > 0 else None
    r2_spring_incr = 1.0 - sse_full / sse_damp if sse_damp > 0 else None

    # -- OQ-5a conditional scale sigma_eps + hour-of-week seasonality ---------
    eidx = np.flatnonzero(fin_e)
    sig_eps2 = np.full(n, np.nan)
    sig_eps2[eidx] = ewm_wall(eps[eidx] ** 2, ts[eidx], HL_EPS_S)
    sig_eps2 = pd.Series(sig_eps2).ffill().to_numpy()
    sigma_eps = np.sqrt(sig_eps2)             # [nats/s^2]

    how = (ts.dayofweek.to_numpy() * 24 + ts.hour.to_numpy()).astype(np.int64)
    s_season = np.ones(n)
    z1_ok = fin_e & (sig_eps2 > 0)
    zi = np.flatnonzero(z1_ok)
    if zi.size:
        dfv = pd.DataFrame({"how": how[zi], "z1": eps[zi] ** 2 / sig_eps2[zi]})
        g = dfv.groupby("how")["z1"]
        csum_excl = g.cumsum() - dfv["z1"]            # strictly prior in-bin sum
        ccnt = dfv.groupby("how").cumcount()
        osum_excl = dfv["z1"].cumsum() - dfv["z1"]    # strictly prior overall
        ocnt = np.arange(len(dfv))
        with np.errstate(divide="ignore", invalid="ignore"):
            s_raw = csum_excl.to_numpy() / ccnt.to_numpy()
            omean = np.where(ocnt >= 1000, osum_excl.to_numpy() / ocnt, np.nan)
            s_val = np.where((ccnt.to_numpy() >= SEASON_MIN_COUNT)
                             & np.isfinite(s_raw) & np.isfinite(omean)
                             & (omean > 0),
                             s_raw / omean, 1.0)
        s_season[zi] = np.clip(s_val, SEASON_CLIP[0], SEASON_CLIP[1])

    # standardized innovations for the Student-t fit
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(z1_ok, eps / (sigma_eps * np.sqrt(s_season)), np.nan)
    zfin = z[np.isfinite(z)]
    tfit = fit_student_t(zfin) if zfin.size > 10_000 else {}
    if tfit:
        # A fitted nu <= 2 has infinite variance and cannot be standardized.
        # Preserve the empirical fit, but make the simulation convention finite
        # and auditable rather than silently using an undefined noise scale.
        nu_fit = tfit["nu"]
        nu_sim = max(nu_fit, NU_SIM_MIN)
        tfit["nu_for_simulation"] = float(nu_sim)
        tfit["nu_floor_applied"] = bool(nu_fit < NU_SIM_MIN)
        tfit["unit_variance_scale_for_simulation"] = float(
            math.sqrt((nu_sim - 2.0) / nu_sim)
        )

    # final causal hour-of-week profile (state as of end of sample)
    season_profile = {}
    if zi.size:
        prof = (pd.Series(dfv["z1"].to_numpy(), index=dfv["how"].to_numpy())
                .groupby(level=0).mean())
        prof = prof / prof.mean()
        season_profile = {
            "min": float(prof.min()), "max": float(prof.max()),
            "argmax_hour_of_week": int(prof.idxmax()),
            "argmin_hour_of_week": int(prof.idxmin()),
        }

    # -- residual diagnostics --------------------------------------------------
    eps_s = pd.Series(eps[fin_e])
    F_s = pd.Series(F[np.isfinite(F)])
    diag = {
        "n_resid": int(fin_e.sum()),
        "std_eps": float(eps_s.std()),
        "std_F": float(F_s.std()),
        "F_distribution": {
            "n": int(F_s.size),
            "mean": float(F_s.mean()),
            "std": float(F_s.std()),
            "skew": float(F_s.skew()),
            "kurtosis": float(F_s.kurt()),
            "percentiles": pctls(F_s.to_numpy()),
        },
        "mean_over_std_eps": float(eps_s.mean() / eps_s.std()),
        "skew_eps": float(eps_s.skew()),
        "kurtosis_eps": float(eps_s.kurt()),
        "std_z": float(np.std(zfin)) if zfin.size else None,
        "kurtosis_z": float(pd.Series(zfin).kurt()) if zfin.size else None,
        "student_t_fit": tfit,
        "acf_eps": {str(L): acf_gap_aware(eps, epoch, L) for L in ACF_LAGS},
        "acf_abs_eps": {str(L): acf_gap_aware(np.abs(eps), epoch, L)
                        for L in (1, 10, 60)},
    }

    # -- daily-sampled parameter stream ---------------------------------------
    prow = np.isfinite(kappa) & obs_valid
    pidx = np.flatnonzero(prow)
    day = epoch[pidx] // 86_400
    last = pidx[np.flatnonzero(np.diff(day, append=day[-1] + 1) != 0)] \
        if pidx.size else np.array([], dtype=np.int64)
    out = pd.DataFrame({
        "timestamp": ts[last],
        "x": x[last], "x_eq": x_eq[last], "disp": d[last],
        "sigma_1min": sigma_hat[last], "m": m[last],
        "kappa": kappa[last], "gamma": gamma[last],
        "k": k[last], "c": c[last],
        "kappa_dt2": kappa[last] * DT_NOM ** 2, "gamma_dt": gamma[last] * DT_NOM,
        "k_dt2": k[last] * DT_NOM ** 2, "c_dt": c[last] * DT_NOM,
        "sigma_eps": sigma_eps[last],
        "n_valid_window": nv[last].astype(np.int64),
    })
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"dynamics_params_{pair}.parquet"
    out.to_parquet(out_path, index=False, compression="zstd")

    summary = {
        "pair": pair,
        "rows": int(n),
        "span": [str(ts[0]), str(ts[-1])],
        "frac_obs_valid": float(obs_valid.mean()),
        "first_param_ts": str(ts[pidx[0]]) if pidx.size else None,
        "n_daily_rows": int(len(out)),
        "params": {
            "m": pctls(m), "sigma_1min": pctls(sigma_hat),
            "kappa": pctls(kappa), "gamma": pctls(gamma),
            "k": pctls(k), "c": pctls(c),
            "kappa_dt2": pctls(kappa * DT_NOM ** 2),
            "gamma_dt": pctls(gamma * DT_NOM),
            "k_dt2": pctls(k * DT_NOM ** 2),
            "c_dt": pctls(c * DT_NOM),
            "sigma_eps": pctls(sigma_eps),
        },
        "frac_kappa_neg": float(np.mean(kappa[np.isfinite(kappa)] < 0))
        if np.isfinite(kappa).any() else None,
        "frac_gamma_neg": float(np.mean(gamma[np.isfinite(gamma)] < 0))
        if np.isfinite(gamma).any() else None,
        "frac_m_clipped": frac_m_clipped,
        "r2_uncentered_full": r2_full,
        "r2_uncentered_damping_only": r2_damp,
        "r2_spring_incremental": r2_spring_incr,
        "residual": diag,
        "seasonality_hour_of_week": season_profile,
        "runtime_s": round(time.time() - t0, 1),
        "output": str(out_path.relative_to(ROOT)),
    }

    # -- OQ-2 sensitivity: refit kappa for alternative x_eq half-lives --------
    if sensitivity:
        sens = {}
        for hl_h in (4, 120):
            xe = ewm_wall(x, ts, hl_h * 3600.0)
            dl = shift1(x - xe)
            sb1, _sb2, sb2o, _ = rolling_ols_2reg(a, dl, vlag, obs_valid,
                                                  OLS_WINDOW, MIN_OBS)
            kap = -sb1
            ee = np.where(obs_valid & np.isfinite(kap),
                          a + kap * dl + (-_sb2) * vlag, np.nan)
            ff = np.isfinite(ee)
            sseF = float(np.nansum(ee[ff] ** 2))
            eeD = np.where(obs_valid & np.isfinite(sb2o),
                           a + (-sb2o) * vlag, np.nan)
            sseD = float(np.nansum(eeD[ff] ** 2))
            sens[f"hl_{hl_h}h"] = {
                "kappa_median": float(np.nanmedian(kap)),
                "kappa_dt2_median": float(np.nanmedian(kap) * DT_NOM ** 2),
                "frac_kappa_neg": float(np.mean(kap[np.isfinite(kap)] < 0)),
                "r2_spring_incremental": 1.0 - sseF / sseD if sseD > 0 else None,
            }
        summary["sensitivity_x_eq_halflife"] = sens

    return summary


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pairs", nargs="+", default=PAIRS_DEFAULT,
                    choices=PAIRS_ALL, metavar="PAIR")
    ap.add_argument("--all", action="store_true", help="run all 10 pairs")
    ap.add_argument("--sensitivity", action="store_true",
                    help="x_eq half-life sensitivity refits (adds ~2x runtime/pair)")
    args = ap.parse_args(argv)
    pairs = PAIRS_ALL if args.all else args.pairs

    OUT_DIR.mkdir(exist_ok=True)
    summary_path = OUT_DIR / "dynamics_summary.json"
    # A summary describes exactly one execution.  Do not retain pairs or a
    # version from a prior invocation: that would mix incompatible estimates.
    combined = {
        "estimator_version": ESTIMATOR_VERSION,
        "run": {
            "pairs": pairs,
            "sensitivity_x_eq_halflife": bool(args.sensitivity),
        },
        "constants": {
        "DT_NOM_s": DT_NOM, "SIGMA_STAR_nats": SIGMA_STAR, "M_CLIP": M_CLIP,
        "HL_EQ_s": HL_EQ_S, "HL_VOL_s": HL_VOL_S, "HL_EPS_s": HL_EPS_S,
        "OLS_WINDOW_bars": OLS_WINDOW, "MIN_OBS": MIN_OBS,
        "SEASON_BINS": SEASON_BINS, "SEASON_MIN_COUNT": SEASON_MIN_COUNT,
        "SEASON_CLIP": SEASON_CLIP, "NU_SIM_MIN": NU_SIM_MIN,
        },
        "pairs": {},
    }

    for pair in pairs:
        print(f"[{pair}] estimating ...", flush=True)
        s = process_pair(pair, sensitivity=args.sensitivity)
        combined["pairs"][pair] = s
        print(f"[{pair}] done in {s['runtime_s']}s -> {s['output']}", flush=True)

    combined["generated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    summary_path.write_text(json.dumps(combined, indent=2))
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Experiment runner.

Each function solves one experiment and caches its result to results/*.json so
that outputs.py can regenerate every table and figure without re-solving. The
main matrix and the perfect-information stages use the Gurobi model; the
independent audit at the end re-solves the convex stages with CVXPY and Clarabel.

Run `python experiments.py` to regenerate all cached results (requires a licensed
Gurobi and, for the audit, cvxpy and clarabel).
"""
from __future__ import annotations
import json, os, time
import numpy as np
import config as C
from data import build_pilot_data, PilotData, annual_summary
from model import (solve_planner, solve_waterfall, price_discovery,
                   _independent, _kpis, _joint, _agent_lp,
                   CARBON_PRICE, DH_PRICE, EPS_SPILL, EPS_DUMP)
import model as M

try:
    import cvxpy as cp
except ImportError:
    cp = None

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [float(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def _slim(r):
    return {k: v for k, v in r.items() if k not in ("series",)}


def _save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(_clean(obj), f, indent=1)
    print(f"  saved {name}")


def scale_storage(d, factor):
    for a in d.agents.values():
        a["batt_kwh"] *= factor; a["batt_kw"] *= factor
        a["tes_kwh"] *= factor; a["tes_kw"] *= factor
    return d


def run_main():
    print("[main] full matrix over all configurations")
    out = {}
    for pid in C.PILOTS:
        t0 = time.time()
        d = build_pilot_data(pid)
        wf, pr = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        splp = solve_planner(d, relax=True)
        pi3, tau3 = pr["DE_M3"]
        _, _, _, pa_m3 = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
        vol3 = sum(pa_m3[a]["sell"] for a in d.agents)     # aggregate pool volume per hour [kW]
        tau_bind = tau3 > 1e-4
        tau_cond = float(np.mean(tau3[tau_bind])) if tau_bind.any() else 0.0
        w = d.days_weight * d.dt
        tau_bind_annual = float(np.sum(w[tau_bind]))
        out[pid] = dict(
            tier=C.PILOTS[pid].tier, country=C.PILOTS[pid].country, zone=C.PILOTS[pid].zone,
            pv_kwp=sum(d.agents[a]["pv_kwp"] for a in d.agents),
            n_agents=len(d.agents), hosting=d.hosting_limit,
            variants={k: wf[k] for k in wf},
            SP=_slim(sp), SP_rel=_slim(splp),
            pi_mean=float(np.mean(pi3)), tau_mean=float(np.mean(tau3)),
            tau_cond_mean=tau_cond, tau_max=float(np.max(tau3)),
            tau_bind_h=int(round(tau_bind_annual)), tau_bind_rep=int(np.sum(tau_bind)),
            integrality_kEUR=float(sp["cost_kEUR"] - splp["cost_kEUR"]),
            simul_kWh=float(splp.get("simul_kWh", 0.0)),
            series=dict(I=sp["series"]["I"], E=sp["series"]["E"], CURT=sp["series"]["CURT"],
                        PV=sp["series"]["PV"], dem=sp["series"]["dem"], pi=pi3, tau=tau3, vol=vol3),
        )
        print(f"  {pid:10s} done in {time.time()-t0:.1f}s")
    _save("main_results.json", out)
    return out


def run_pareto(pids, fracs=None):
    """Cost of raising hourly self-sufficiency: tighten an annual import cap below
    the unconstrained optimum and record (self-sufficiency, cost)."""
    fracs = fracs or [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.65, 0.60, 0.50, 0.40]
    print("[pareto] self-sufficiency cost curve")
    out = {}
    for pid in pids:
        d = build_pilot_data(pid)
        base = solve_planner(d, relax=True)
        base_imp = base["imp_MWh"] * 1000
        pts = [dict(frac=1.0, cost_kEUR=base["cost_kEUR"], SSR=base["SSR"],
                    co2_t=base["co2_t"], LPI_os=base["LPI_os"])]
        for f in fracs[1:]:
            r = solve_planner(d, import_cap=base_imp * f, relax=True)
            if (r.get("ok") and 0.0 <= r["SSR"] <= 1.0
                    and r["cost_kEUR"] <= 3.0 * base["cost_kEUR"]
                    and r["SSR"] >= base["SSR"] - 1e-3):
                pts.append(dict(frac=f, cost_kEUR=r["cost_kEUR"], SSR=r["SSR"],
                                co2_t=r["co2_t"], LPI_os=r["LPI_os"]))
        out[pid] = pts
        print(f"  {pid} done ({len(pts)} feasible points)")
    _save("pareto.json", out)
    return out


def run_sensitivity(pid="virum"):
    print(f"[sensitivity] pilot={pid}")
    out = {"pid": pid}
    base_host = build_pilot_data(pid).hosting_limit

    hosting = []
    for s in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        d = build_pilot_data(pid); d.hosting_limit = base_host * s
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        hosting.append(dict(scale=s, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  hosting x{s} done")
    out["hosting"] = hosting

    carbon = []
    for cp_ in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        d = build_pilot_data(pid)
        wf, _ = solve_waterfall(d, carbon_price=cp_)
        sp = solve_planner(d, carbon_price=cp_, relax=False)
        carbon.append(dict(carbon=cp_, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  carbon {cp_} done")
    out["carbon"] = carbon

    storage = []
    for s in [0.0, 0.5, 1.0, 2.0, 3.0]:
        d = scale_storage(build_pilot_data(pid), s)
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        storage.append(dict(scale=s, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  storage x{s} done")
    out["storage"] = storage

    res = []
    p = C.PILOTS[pid]; orig_ratio = p.ped_ratio
    for r in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        p.ped_ratio = r
        d = build_pilot_data(pid)
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        res.append(dict(ratio=r, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  RES ratio {r} done")
    p.ped_ratio = orig_ratio
    out["res"] = res

    flex = []
    orig_alpha = C.ALPHA_SHIFT
    for a in [0.0, 0.15, 0.30, 0.45, 0.60]:
        C.ALPHA_SHIFT = a
        d = build_pilot_data(pid)
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        flex.append(dict(alpha=a, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  flex alpha {a} done")
    C.ALPHA_SHIFT = orig_alpha
    out["flex"] = flex

    feedin = []
    for fr in [0.2, 0.4, 0.6, 0.8, 1.0]:
        d = build_pilot_data(pid)
        d.price_exp = fr * d.price_imp
        wf, _ = solve_waterfall(d)
        sp = solve_planner(d, relax=False)
        feedin.append(dict(ratio=fr, DE0=_slim(wf["DE0"]), DE_M3=_slim(wf["DE_M3"]), SP=_slim(sp)))
        print(f"  feed-in {fr} done")
    out["feedin"] = feedin

    _save(f"sensitivity_{pid}.json", out)
    return out


def run_fullyear(pid="virum"):
    """Hourly full-year (8736 h) profile check against the four-representative-week
    model. Each season is solved as a storage-neutral convex block and aggregated."""
    print(f"[fullyear] pilot={pid} (8736 h)")
    orig = C.HOURS_PER_SEASON
    main = json.load(open(os.path.join(RESULTS, "main_results.json")))
    r = main[pid]
    rep = dict(cost_kEUR=r["SP"]["cost_kEUR"], SSR=r["SP"]["SSR"], LPI_os=r["SP"]["LPI_os"],
               curt_MWh=r["SP"]["curt_MWh"], co2_t=r["SP"]["co2_t"],
               gap_kEUR=r["variants"]["DE0"]["cost_kEUR"] - r["SP"]["cost_kEUR"],
               tau_bind_h=r["tau_bind_h"])

    def _season_slice(d, i, hps):
        sl = slice(i * hps, (i + 1) * hps)
        ags = {a: {k: (v[sl].copy() if isinstance(v, np.ndarray) else v) for k, v in ag.items()}
               for a, ag in d.agents.items()}
        return PilotData(pid=d.pid, T=hps, dt=d.dt, season=d.season[sl], days_weight=d.days_weight[sl],
                         Tamb=d.Tamb[sl], price_imp=d.price_imp[sl], price_exp=d.price_exp[sl],
                         tariff=d.tariff[sl], ef=d.ef[sl], ef_dh=d.ef_dh, agents=ags,
                         hosting_limit=d.hosting_limit)

    C.HOURS_PER_SEASON = 2184
    try:
        d = build_pilot_data(pid)
        cost = co2 = curt = enet = inet = dem = pv = bind = 0.0
        for i in range(len(C.SEASONS)):
            ds = _season_slice(d, i, 2184)
            sp = solve_planner(ds, relax=True)
            cost += sp["cost_kEUR"]; co2 += sp["co2_t"]; curt += sp["curt_MWh"]
            enet += sp["exp_MWh"]; inet += sp["imp_MWh"]; dem += sp["dem_MWh"]; pv += sp["pv_MWh"]
            E = np.array(sp["series"]["E"]); w = ds.days_weight * ds.dt
            bind += float(np.sum((E >= 0.99 * ds.hosting_limit) * w))
        a0, _, _ = _independent(d, "de0", CARBON_PRICE, DH_PRICE)
        de0 = _kpis(d, a0, "DE0", d.hosting_limit)
        fy = dict(cost_kEUR=cost, SSR=1 - inet / max(dem, 1e-9),
                  LPI_os=(enet + curt) / max(pv, 1e-9), curt_MWh=curt, co2_t=co2,
                  gap_kEUR=de0["cost_kEUR"] - cost, tau_bind_h=bind)
    finally:
        C.HOURS_PER_SEASON = orig
    _save("fullyear.json", {"pid": pid, "rep_week": rep, "full_year": fy})
    return {"pid": pid, "rep_week": rep, "full_year": fy}


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def _uncert_stats(recs, pid):
    keys = ["cost_SP", "SSR", "gap_kEUR", "gap_pct", "curt_SP",
            "m1_recovery_kEUR", "m1_recovery_pct", "curt_reduction_pct",
            "share_info", "share_foresight", "share_pool", "share_grid",
            "ann_gen_MWh", "ped_ratio"]
    stats = {k: dict(mean=float(np.mean([r[k] for r in recs])),
                     std=float(np.std([r[k] for r in recs])),
                     p10=float(np.percentile([r[k] for r in recs], 10)),
                     p25=float(np.percentile([r[k] for r in recs], 25)),
                     p75=float(np.percentile([r[k] for r in recs], 75)),
                     p90=float(np.percentile([r[k] for r in recs], 90))) for k in keys}
    stats["rank_preserved"] = float(np.mean([r["rank_ok"] for r in recs]))
    stats["n"] = len(recs)
    return {"pid": pid, "stats": stats, "draws": recs}


def _uncert_record(wf, s, d):
    m3 = wf["DE_M3"]                                     # equals SP by the zero-integrality-gap equivalence
    c = {k: wf[k]["cost_kEUR"] for k in ["DE0", "DE_M1", "DE_IND", "DE_M2", "DE_M3"]}
    c["SP"] = c["DE_M3"]
    gap = c["DE0"] - c["SP"]
    curt_de0 = wf["DE0"]["curt_MWh"]; curt_m3 = wf["DE_M3"]["curt_MWh"]
    asum = annual_summary(d)
    ann_gen = asum["pv_MWh"]; ped = ann_gen / max(asum["demand_eq_MWh"], 1e-9)
    return dict(
        seed_idx=s, cost_SP=c["SP"], SSR=m3["SSR"], gap_kEUR=gap, gap_pct=100 * gap / c["DE0"],
        curt_SP=m3["curt_MWh"], ann_gen_MWh=ann_gen, ped_ratio=ped,
        m1_recovery_kEUR=c["DE0"] - c["DE_M1"],
        m1_recovery_pct=100 * (c["DE0"] - c["DE_M1"]) / gap if gap > 1e-6 else 0.0,
        curt_reduction_pct=100 * (curt_de0 - curt_m3) / curt_de0 if curt_de0 > 1e-6 else 0.0,
        share_info=(c["DE0"] - c["DE_M1"]) / gap if gap > 1e-6 else 0.0,
        share_foresight=(c["DE_M1"] - c["DE_IND"]) / gap if gap > 1e-6 else 0.0,
        share_pool=(c["DE_IND"] - c["DE_M2"]) / gap if gap > 1e-6 else 0.0,
        share_grid=(c["DE_M2"] - c["DE_M3"]) / gap if gap > 1e-6 else 0.0,
        rank_ok=int(c["DE0"] >= c["DE_M1"] - 1e-6 >= c["DE_M2"] - 1e-6 >= c["DE_M3"] - 1e-6 >= c["SP"] - 1e-6),
    )


def _uncert_draw(pid, s):
    d = build_pilot_data(pid, seed=C.SEED + 101 * s)
    wf, _ = solve_waterfall(d)
    return _uncert_record(wf, s, d)


def _fix_capacity(d, base_kwp, yield_factor=1.0):
    """Hold installed capacity at the base value; ``yield_factor`` scales the annual
    energy by an interannual weather-year multiplier so annual generation varies
    through the yield while installed capacity stays fixed."""
    for a, v in d.agents.items():
        if v["pv_kwp"] > 1e-9:
            v["PVavail"] = v["PVavail"] * (base_kwp[a] / v["pv_kwp"]) * yield_factor
            v["pv_kwp"] = base_kwp[a]
    return d


def run_uncertainty(pid="virum", seeds=100):
    """Monte-Carlo over weather, demand, and forecast-error seeds. Reports the
    distribution of the gap and the ordered mechanism shares and how often the
    qualitative ranking is preserved."""
    print(f"[uncertainty] pilot={pid} ({seeds} seeds)")
    recs = [_uncert_draw(pid, s) for s in range(seeds)]
    _save("uncertainty.json", _uncert_stats(recs, pid))
    return _uncert_stats(recs, pid)["stats"]


def run_uncertainty_resumable(pid="virum", target=100, budget_s=520):
    """Resumable Monte-Carlo. Draws are accumulated in results/uncertainty_raw.json
    and saved after each draw; the final uncertainty.json is written on reaching the
    target. Call repeatedly to top up under a wall-time budget."""
    raw_path = os.path.join(RESULTS, "uncertainty_raw.json")
    recs = json.load(open(raw_path))["draws"] if os.path.exists(raw_path) else []
    done = {r["seed_idx"] for r in recs}
    t0 = time.time(); s = 0
    while len(recs) < target:
        while s in done:
            s += 1
        if s >= target:
            break
        recs.append(_uncert_draw(pid, s)); done.add(s)
        json.dump(_clean({"pid": pid, "draws": recs}), open(raw_path, "w"), indent=1)
        print(f"  draw {len(recs)}/{target} (seed {s}) at {time.time()-t0:.0f}s")
        s += 1
        if time.time() - t0 > budget_s:
            print(f"  budget reached at {len(recs)} draws; rerun to continue")
            return len(recs)
    _save("uncertainty.json", _uncert_stats(recs, pid))
    print(f"  COMPLETE at {len(recs)} draws")
    return len(recs)


# --------------------------------------------------------------------------
# Robustness of the deployable step and the no-platform baseline
# --------------------------------------------------------------------------
def run_forecast(pid="virum"):
    """Sensitivity of the information step DE0->M1 to the assumed day-ahead
    forecast skill. The private error is swept with the shared error held at base."""
    print(f"[forecast] pilot={pid}")
    orig_p, orig_s = M.ERR_PRIV, M.ERR_SHARED
    d = build_pilot_data(pid)
    sp = solve_planner(d, relax=False)["cost_kEUR"]
    rows = []
    try:
        for ep in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
            M.ERR_PRIV = ep; M.ERR_SHARED = orig_s
            wf, _ = solve_waterfall(d)
            de0 = wf["DE0"]["cost_kEUR"]; m1 = wf["DE_M1"]["cost_kEUR"]
            gap = de0 - sp
            rows.append(dict(err_priv=ep, err_shared=orig_s, de0_kEUR=de0, m1_kEUR=m1,
                             info_kEUR=de0 - m1,
                             info_pct_gap=100 * (de0 - m1) / gap if gap > 1e-6 else 0.0,
                             gap_kEUR=gap))
            print(f"  err_priv {ep} done")
    finally:
        M.ERR_PRIV, M.ERR_SHARED = orig_p, orig_s
    _save("forecast.json", {"pid": pid, "sp_kEUR": sp, "base_priv": orig_p,
                            "base_shared": orig_s, "rows": rows})
    return rows


def run_baseline(pid="virum"):
    """Robustness of the DE0 baseline to a 48 h rolling horizon and a
    mean-preserving time-of-use network tariff."""
    print(f"[baseline] pilot={pid}")
    H = build_pilot_data(pid).hosting_limit
    sp = solve_planner(build_pilot_data(pid), relax=False)["cost_kEUR"]

    def _de0(d, horizon):
        a0, _, _ = _independent(d, "de0", CARBON_PRICE, DH_PRICE, horizon=horizon)
        return _kpis(d, a0, "DE0", H)["cost_kEUR"]

    base = _de0(build_pilot_data(pid), 24)
    h48 = _de0(build_pilot_data(pid), 48)
    d_tou = build_pilot_data(pid)
    hod = np.arange(d_tou.T) % 24
    fac = np.ones(d_tou.T)
    fac[(hod >= 17) & (hod <= 21)] = 1.6
    fac[(hod >= 0) & (hod <= 5)] = 0.6
    base_mean = float(np.mean(d_tou.tariff))
    d_tou.tariff = d_tou.tariff * fac
    d_tou.tariff = d_tou.tariff * (base_mean / max(float(np.mean(d_tou.tariff)), 1e-9))
    tou = _de0(d_tou, 24)
    g0 = base - sp
    rows = [dict(case="base_24h_flat", de0_kEUR=base, gap_kEUR=g0),
            dict(case="horizon_48h", de0_kEUR=h48, gap_kEUR=h48 - sp),
            dict(case="time_of_use", de0_kEUR=tou, gap_kEUR=tou - sp)]
    for r in rows:
        r["gap_pct_of_base"] = 100 * r["gap_kEUR"] / g0 if abs(g0) > 1e-9 else 0.0
    _save("baseline.json", {"pid": pid, "sp_kEUR": sp, "rows": rows})
    return rows


# --------------------------------------------------------------------------
# Market power and individual rationality
# --------------------------------------------------------------------------
def _strategic(pid):
    d = build_pilot_data(pid); w = d.days_weight * d.dt; A = list(d.agents.keys())
    _, pi, tau, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
    L = max(A, key=lambda a: d.agents[a]["pv_kwp"])
    net = {a: pa[a]["exp"] - pa[a]["imp"] for a in A}
    agg_net = sum(net.values())
    bind = tau > 1e-6
    wb = w[bind]
    denom = float(np.sum(wb * np.maximum(agg_net, 0.0)[bind]))

    def _share(x):
        return float(np.sum(wb * np.maximum(x, 0.0)[bind])) / denom if denom > 1e-9 else 0.0

    cong_L = float(np.sum(w * tau * net[L])) / 1000
    bill_L = pa[L]["bill"]
    return dict(pid=pid, n_agents=len(A), largest=L,
                pv_share=float(d.agents[L]["pv_kwp"] / sum(d.agents[a]["pv_kwp"] for a in A)),
                export_share_binding=_share(net[L]),
                hhi=float(np.sum(np.square([_share(net[a]) for a in A]))),
                cong_payment_kEUR=cong_L, bill_kEUR=bill_L,
                dev_gain_bound_pct=100 * max(cong_L, 0.0) / bill_L if bill_L > 1e-9 else 0.0,
                bind_hours_rep=int(np.sum(bind)))


def run_marketpower(pids=None):
    """Market-power indicators per configuration: binding-hour net-export
    Herfindahl index, largest-agent export share, its congestion payment, and a
    loose upper bound on any unilateral withholding gain."""
    pids = pids or list(C.PILOTS)
    print(f"[marketpower] {pids}")
    rows = [_strategic(pid) for pid in pids]
    _save("marketpower.json", {"rows": rows})
    return rows


def run_pool_price(pids=None):
    """P2P clearing-price statistics that respect pool-dual non-uniqueness: the
    trade-volume-weighted mean price over active hours, the active-hour count, the
    trade-weighted seller premium above feed-in and buyer saving below the effective
    import price, and the all-hour arithmetic mean (the particular solver dual)."""
    pids = pids or list(C.PILOTS)
    print(f"[pool-price] {pids}")
    rows = []
    for pid in pids:
        d = build_pilot_data(pid); A = list(d.agents.keys()); w = d.days_weight * d.dt
        _, pi, tau, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
        vol = sum(pa[a]["sell"] for a in A)
        act = vol > 1.0
        wv = w * vol
        tw = lambda x: float(np.sum(wv[act] * x[act]) / np.sum(wv[act])) if act.any() else 0.0
        # objective-based M3 grid alternatives from model (4): use-of-system tariff on
        # both directions, congestion price on net export, monetized carbon on import
        c_imp = d.price_imp + d.tariff + CARBON_PRICE * d.ef - tau
        r_exp = d.price_exp - d.tariff - tau
        rows.append(dict(pid=pid, pi_tw=tw(pi), n_active_h=float(np.sum(w[act])),
                         n_active_rep=int(np.sum(act)), seller_prem=tw(pi - r_exp),
                         buyer_save=tw(c_imp - pi), pi_all_solverdual=float(np.mean(pi))))
    _save("pool_price.json", {"rows": rows})
    return rows


def run_agent_table(pid="virum"):
    """Per-agent operating cost under DE0 and M3. An agent is individually rational
    to join only if its M3 bill (which carries the unrecycled congestion payment)
    does not exceed its DE0 outside option."""
    print(f"[agent-table] pilot={pid}")
    d = build_pilot_data(pid); A = list(d.agents.keys()); T = d.T
    w = d.days_weight * d.dt; H = d.hosting_limit
    rng = np.random.default_rng(C.SEED + 0)
    keys = ["imp", "exp", "PV", "chB", "disB", "DH"]
    ser = {a: {k: np.zeros(T) for k in keys} for a in A}
    windows = list(range(0, T, 24)); nwin = len(windows); SL = C.HOURS_PER_SEASON
    for a in A:
        ag = d.agents[a]
        ref = {"eB": 0.5 * ag["batt_kwh"], "eT": 0.5 * ag["tes_kwh"]}
        e0 = dict(ref)
        for wi, d0 in enumerate(windows):
            steps = list(range(d0, min(d0 + 24, T))); end = min(d0 + 24, T)
            sl = slice(d0, end); L = len(steps)
            if d0 % SL == 0:
                e0 = dict(ref)
            pv_t = ag["PVavail"][sl]; df_t = ag["Dfix"][sl]; ds_t = ag["Dshbase"][sl]; dh_t = ag["Dheat"][sl]
            fe = {q: np.clip(1 + rng.normal(0, M.ERR_PRIV, L), 0.2, None) for q in ["pv", "load", "heat"]}
            last_of_week = (end % SL == 0 or wi == nwin - 1)
            terminal = dict(ref) if last_of_week else None
            close = {"eT": ref["eT"]} if last_of_week else None
            plan = _agent_lp(d, a, steps, pv_t * fe["pv"], df_t * fe["load"],
                             ds_t * fe["load"], dh_t * fe["heat"], e0, False, CARBON_PRICE, DH_PRICE, terminal=terminal)
            real = _agent_lp(d, a, steps, pv_t, df_t, ds_t, dh_t, e0, False, CARBON_PRICE, DH_PRICE, fix=plan, terminal=close)
            for k in keys:
                ser[a][k][sl] += real[k]
            e0 = {"eB": real["eB_end"], "eT": real["eT_end"]}
    agg = {k: sum(ser[a][k] for a in A) for k in keys}
    net = agg["imp"] - agg["exp"]
    excess = np.maximum(np.maximum(-net, 0.0) - H, 0.0)
    curt_host = np.minimum(excess, agg["PV"])
    share = np.where(agg["exp"] > 1e-9, curt_host / np.maximum(agg["exp"], 1e-9), 0.0)
    de0 = {}
    for a in A:
        s = ser[a]
        exp_eff = np.maximum(s["exp"] - s["exp"] * share, 0.0)
        de0[a] = float(np.sum(w * (d.price_imp * s["imp"] - d.price_exp * exp_eff
                        + d.tariff * (s["imp"] + exp_eff) + C.DEG_COST * (s["chB"] + s["disB"])
                        + DH_PRICE * s["DH"] + CARBON_PRICE * (d.ef * s["imp"] + d.ef_dh * s["DH"])))) / 1000
    _, _, _, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
    m3 = {a: pa[a]["bill"] for a in A}
    rows = [dict(agent=a, de0_kEUR=de0[a], m3_kEUR=m3[a], delta_kEUR=m3[a] - de0[a],
                 ir_ok=int(m3[a] <= de0[a] + 1e-6)) for a in A]
    tot_de0 = sum(de0.values()); tot_m3 = sum(m3.values())
    _save("agent_table.json", dict(pid=pid, rows=rows, tot_de0=tot_de0, tot_m3=tot_m3,
                                   community_save_kEUR=tot_de0 - tot_m3,
                                   n_ir=int(sum(r["ir_ok"] for r in rows)), n_agents=len(A)))
    return rows


def run_convergence(pid="virum"):
    """Diagnostic for the illustrative dual-decomposition price discovery. Not used
    for any reported value."""
    print(f"[convergence] pilot={pid}")
    d = build_pilot_data(pid)
    res = price_discovery(d, iters=250, gamma0=0.04)
    _, pr = solve_waterfall(d)
    pi_c, tau_c = pr["DE_M3"]
    _save("convergence.json", dict(
        pid=pid, hist=res["hist"],
        pi_err=float(np.sqrt(np.mean((res["pi"] - pi_c) ** 2))),
        tau_err=float(np.sqrt(np.mean((res["tau"] - tau_c) ** 2))),
        pi_c_mean=float(np.mean(pi_c)), tau_c_mean=float(np.mean(tau_c)),
        pi_d_mean=float(np.mean(res["pi"])), tau_d_mean=float(np.mean(res["tau"])),
        normalization=res["normalization"]))


# --------------------------------------------------------------------------
# Independent convex-solver audit (CVXPY / Clarabel)
#
# Reproduces the convex perfect-information stages for the primary configuration
# independently of the Gurobi implementation, and reports the regularizer
# sensitivity and the pool/grid attribution-order audit.
# --------------------------------------------------------------------------
def _cvx_bounds(ag):
    flow = 5.0 * (float(np.max(ag["Dfix"] + ag["Dshbase"])) + float(np.max(ag["PVavail"])) + float(ag["batt_kw"]) + 1.0)
    shift = C.W_SHIFT * float(np.mean(ag["Dshbase"])) if ag["flex"] else 0.0
    return flow, shift


def _cvx_vars(T):
    names = ["PV", "chB", "disB", "chT", "disT", "Phe", "Pres", "Qd", "DH", "dsh", "imp", "exp", "buy", "sell"]
    v = {n: cp.Variable(T, nonneg=True) for n in names}
    v["eB"] = cp.Variable(T + 1); v["eT"] = cp.Variable(T + 1); v["qsh"] = cp.Variable(T + 1)
    return v


def _cvx_agent_constraints(d, aid, v, pool_enabled):
    ag = d.agents[aid]; T, dt = d.T, d.dt
    flow, shift = _cvx_bounds(ag)
    cop = np.clip(C.COP0 + C.KAPPA_COP * (d.Tamb - 7.0), C.COP_MIN, C.COP_MAX)
    served_cap = 2.0 * ag["Dshbase"] + 1e-6 if ag["flex"] else ag["Dshbase"]
    cons = [
        v["PV"] <= ag["PVavail"], v["chB"] <= ag["batt_kw"], v["disB"] <= ag["batt_kw"],
        v["chT"] <= ag["tes_kw"], v["disT"] <= ag["tes_kw"],
        v["Phe"] <= (2e4 if ag["hp"] else 0.0), v["Pres"] <= 2e4, v["Qd"] <= ag["Qpvt"],
        v["DH"] <= (2e4 if ag["dh"] else 0.0), v["dsh"] <= served_cap,
        v["imp"] <= flow, v["exp"] <= flow,
        v["buy"] <= (flow if pool_enabled else 0.0), v["sell"] <= (flow if pool_enabled else 0.0),
        v["eB"] >= C.SOC_MIN * ag["batt_kwh"], v["eB"] <= max(C.SOC_MAX * ag["batt_kwh"], 1e-6),
        v["eT"] >= 0.0, v["eT"] <= max(ag["tes_kwh"], 1e-6),
        v["qsh"] >= -shift, v["qsh"] <= shift,
        v["PV"] + v["disB"] + v["imp"] + v["buy"]
        == ag["Dfix"] + v["dsh"] + v["chB"] + v["Phe"] + v["Pres"] + v["exp"] + v["sell"],
        cp.multiply(cop, v["Phe"]) + v["Pres"] + ag["Qpvt"] + v["disT"] + v["DH"] == ag["Dheat"] + v["chT"] + v["Qd"],
        v["eB"][1:] == v["eB"][:-1] + C.ETA_CH * v["chB"] * dt - v["disB"] * dt / C.ETA_DIS,
        v["eT"][1:] == (1.0 - C.KAPPA_TES * dt) * v["eT"][:-1] + C.ETA_TES * v["chT"] * dt - v["disT"] * dt / C.ETA_TES,
        v["qsh"][1:] == v["qsh"][:-1] + ag["Dshbase"] - v["dsh"],
    ]
    if ag["flex"]:
        cap = C.ALPHA_SHIFT * (ag["Dfix"] + ag["Dshbase"])
        cons += [v["dsh"] - ag["Dshbase"] <= cap, ag["Dshbase"] - v["dsh"] <= cap]
    for b in range(C.HOURS_PER_SEASON, T + 1, C.HOURS_PER_SEASON):
        cons += [v["eB"][b] == v["eB"][0], v["eT"][b] == v["eT"][0]]
    for b in range(0, T + 1, 24):
        cons.append(v["qsh"][b] == 0.0)
    return cons


def _cvx_objective(d, aid, v, regularization):
    ag = d.agents[aid]; w = d.days_weight * d.dt
    economic = (cp.multiply(d.price_imp, v["imp"]) - cp.multiply(d.price_exp, v["exp"])
                + cp.multiply(d.tariff, v["imp"] + v["exp"]) + C.DEG_COST * (v["chB"] + v["disB"])
                + DH_PRICE * v["DH"] + CARBON_PRICE * (cp.multiply(d.ef, v["imp"]) + d.ef_dh * v["DH"]))
    numerical = (EPS_SPILL * (ag["PVavail"] - v["PV"]) + EPS_DUMP * v["Qd"]
                 + regularization * (cp.square(v["imp"]) + cp.square(v["exp"]) + cp.square(v["buy"]) + cp.square(v["sell"])))
    return cp.sum(cp.multiply(w, economic + numerical))


def _cvx_solve_joint(d, pool_enabled, hosting_internalized, regularization, solver="CLARABEL"):
    A = list(d.agents); T = d.T
    var = {a: _cvx_vars(T) for a in A}
    cons = []; obj = 0.0
    for a in A:
        cons += _cvx_agent_constraints(d, a, var[a], pool_enabled)
        obj = obj + _cvx_objective(d, a, var[a], regularization)
    pool_c, host_c = [], []
    for t in range(T):
        if pool_enabled:
            c = sum(var[a]["sell"][t] - var[a]["buy"][t] for a in A) == 0.0
            cons.append(c); pool_c.append(c)
        if hosting_internalized:
            c = sum(var[a]["exp"][t] - var[a]["imp"][t] for a in A) <= d.hosting_limit
            cons.append(c); host_c.append(c)
    scale = 1000.0
    prob = cp.Problem(cp.Minimize(scale * obj), cons)
    prob.solve(solver=solver, verbose=False)
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"convex audit joint solve failed: {prob.status}")
    keys = ["imp", "exp", "PV", "chB", "disB", "DH", "Phe", "Pres", "dsh", "buy", "sell"]
    arrays = {k: sum(np.asarray(var[a][k].value, dtype=float) for a in A) for k in keys}
    w = d.days_weight * d.dt
    pi = (np.array([-float(c.dual_value) / (scale * max(w[t], 1e-12)) for t, c in enumerate(pool_c)])
          if pool_enabled else None)
    tau = (np.array([float(c.dual_value) / (scale * max(w[t], 1e-12)) for t, c in enumerate(host_c)])
           if hosting_internalized else np.zeros(T))
    return dict(raw=float(prob.value) / scale, arrays=arrays, pi=pi, tau=tau)


def _cvx_solve_independent(d, regularization, solver="CLARABEL"):
    T = d.T
    keys = ["imp", "exp", "PV", "chB", "disB", "DH", "Phe", "Pres", "dsh", "buy", "sell"]
    agg = {k: np.zeros(T) for k in keys}
    raw = 0.0
    for a in d.agents:
        v = _cvx_vars(T)
        prob = cp.Problem(cp.Minimize(1000.0 * _cvx_objective(d, a, v, regularization)),
                          _cvx_agent_constraints(d, a, v, pool_enabled=False))
        prob.solve(solver=solver, verbose=False)
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"convex audit independent solve failed for {a}: {prob.status}")
        raw += float(prob.value) / 1000.0
        for k in agg:
            agg[k] += np.asarray(v[k].value, dtype=float)
    return dict(raw=raw, arrays=agg, pi=None, tau=np.zeros(T))


def _cvx_metrics(d, sol, apply_hosting_recourse):
    a = sol["arrays"]; w = d.days_weight * d.dt
    net = a["imp"] - a["exp"]
    excess = np.maximum(np.maximum(-net, 0.0) - d.hosting_limit, 0.0) if apply_hosting_recourse else np.zeros(d.T)
    curtailed = np.minimum(excess, a["PV"])
    exp_eff = np.maximum(a["exp"] - curtailed, 0.0)
    operating = np.sum(w * (d.price_imp * a["imp"] - d.price_exp * exp_eff + d.tariff * (a["imp"] + exp_eff)
                            + C.DEG_COST * (a["chB"] + a["disB"]) + DH_PRICE * a["DH"]
                            + CARBON_PRICE * (d.ef * a["imp"] + d.ef_dh * a["DH"])))
    return {
        "economic_cost_kEUR": float(operating / 1000.0),
        "raw_pre_recourse_objective_kEUR": float(sol["raw"] / 1000.0),
        "pool_volume_MWh": float(np.sum(w * a["buy"]) / 1000.0),
        "pi_mean_EUR_per_kWh": float(np.mean(sol["pi"])) if sol["pi"] is not None else 0.0,
        "tau_mean_EUR_per_kWh": float(np.mean(sol["tau"])) if sol["tau"] is not None else 0.0,
        "tau_max_EUR_per_kWh": float(np.max(sol["tau"])) if sol["tau"] is not None else 0.0,
    }


def run_convex_audit(pid="virum"):
    if cp is None:
        raise ImportError("cvxpy is required for the independent audit (pip install cvxpy clarabel)")
    print(f"[convex-audit] pilot={pid}")
    d = build_pilot_data(pid)
    reg_rows = []
    for eps in [1e-7, 1e-6, 1e-5]:
        ind = _cvx_metrics(d, _cvx_solve_independent(d, eps), apply_hosting_recourse=True)
        m2 = _cvx_metrics(d, _cvx_solve_joint(d, True, False, eps), apply_hosting_recourse=True)
        m3 = _cvx_metrics(d, _cvx_solve_joint(d, True, True, eps), apply_hosting_recourse=False)
        reg_rows.append(dict(epsilon=eps, IND=ind, M2=m2, M3=m3,
                             pool_increment_kEUR=ind["economic_cost_kEUR"] - m2["economic_cost_kEUR"],
                             grid_increment_kEUR=m2["economic_cost_kEUR"] - m3["economic_cost_kEUR"]))
        print(f"  regularizer {eps:g} done")
    reg = 1e-6
    C0 = _cvx_metrics(d, _cvx_solve_independent(d, reg), apply_hosting_recourse=True)["economic_cost_kEUR"]
    Cp = _cvx_metrics(d, _cvx_solve_joint(d, True, False, reg), apply_hosting_recourse=True)["economic_cost_kEUR"]
    Cg = _cvx_metrics(d, _cvx_solve_joint(d, False, True, reg), apply_hosting_recourse=False)["economic_cost_kEUR"]
    Cpg = _cvx_metrics(d, _cvx_solve_joint(d, True, True, reg), apply_hosting_recourse=False)["economic_cost_kEUR"]
    order = dict(
        regularization=reg,
        pool_first={"pool_kEUR": C0 - Cp, "grid_kEUR": Cp - Cpg},
        grid_first={"grid_kEUR": C0 - Cg, "pool_kEUR": Cg - Cpg},
        shapley={"pool_kEUR": 0.5 * ((C0 - Cp) + (Cg - Cpg)), "grid_kEUR": 0.5 * ((C0 - Cg) + (Cp - Cpg))},
        interaction_kEUR=C0 - Cp - Cg + Cpg,
    )
    _save("convex_audit.json", dict(
        description="Independent CVXPY/Clarabel audit of the convex perfect-information stages.",
        regularization_sensitivity=reg_rows, order_audit=order))


def run_reg_deployable(pid="virum"):
    """Regularizer sensitivity of the deployable stages DE0 and M1. The costs and
    the information recovery should be stable across the coupling-flow regularizer."""
    print(f"[reg-deployable] pilot={pid}")
    d = build_pilot_data(pid)
    rows = []
    for eps in [1e-7, 1e-6, 1e-5]:
        wf, _ = solve_waterfall(d, regularization=eps)
        de0 = wf["DE0"]["cost_kEUR"]; m1 = wf["DE_M1"]["cost_kEUR"]; gap = de0 - wf["DE_M3"]["cost_kEUR"]
        rows.append(dict(epsilon=eps, de0_kEUR=de0, m1_kEUR=m1, info_kEUR=de0 - m1,
                         m1_recovery_pct=100 * (de0 - m1) / gap if gap > 1e-6 else 0.0, gap_kEUR=gap))
    _save("regularization_deployable.json", rows)
    return rows


def run_uncertainty_fixed(pid="virum", seeds=100):
    """Fixed-capacity Monte Carlo: installed PV held at the base value while the
    annual yield varies with a per-draw interannual weather-year multiplier, so
    annual generation varies through the yield rather than through capacity.
    Complements the re-sized study as a genuine fixed-district robustness test."""
    print(f"[uncertainty-fixed] pilot={pid} ({seeds} seeds)")
    base_kwp = {a: v["pv_kwp"] for a, v in build_pilot_data(pid).agents.items()}
    recs = []
    for s in range(seeds):
        yf = float(np.clip(np.random.default_rng(C.SEED + 101 * s + 7).normal(1.0, 0.05), 0.85, 1.15))
        d = _fix_capacity(build_pilot_data(pid, seed=C.SEED + 101 * s), base_kwp, yf)
        wf, _ = solve_waterfall(d)
        recs.append(_uncert_record(wf, s, d))
    _save("uncertainty_fixed.json", _uncert_stats(recs, pid))
    return _uncert_stats(recs, pid)["stats"]


def run_recycling(pid="virum"):
    """Congestion-revenue recycling and individual rationality. Returns each
    agent's M3 bill after equal per-capita and payment-proportional recycling and
    whether individual rationality is restored relative to DE0. Requires agent_table.json."""
    print(f"[recycling] pilot={pid}")
    d = build_pilot_data(pid); w = d.days_weight * d.dt; A = list(d.agents.keys())
    _, pi, tau, pa = _joint(d, True, CARBON_PRICE, DH_PRICE, detail=True)
    m3 = {a: pa[a]["bill"] for a in A}
    cong = {a: float(np.sum(w * tau * (pa[a]["exp"] - pa[a]["imp"]))) / 1000 for a in A}
    R = sum(cong.values()); n = len(A)
    de0 = {r["agent"]: r["de0_kEUR"] for r in json.load(open(os.path.join(RESULTS, "agent_table.json")))["rows"]}
    rows = []
    for a in A:
        eq = m3[a] - R / n
        prop = m3[a] - cong[a]
        rows.append(dict(agent=a, de0=de0[a], m3=m3[a], cong=cong[a], m3_eq=eq, m3_prop=prop,
                         ir_m3=int(m3[a] <= de0[a] + 1e-6), ir_eq=int(eq <= de0[a] + 1e-6),
                         ir_prop=int(prop <= de0[a] + 1e-6)))
    _save("recycling.json", dict(pid=pid, revenue_kEUR=R, n=n, rows=rows,
                                 n_ir_m3=sum(r["ir_m3"] for r in rows),
                                 n_ir_eq=sum(r["ir_eq"] for r in rows),
                                 n_ir_prop=sum(r["ir_prop"] for r in rows)))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    run_main()
    run_pareto(C.DEEP_PILOTS)
    run_sensitivity("virum")
    run_uncertainty("virum", seeds=100)
    run_uncertainty_fixed("virum", seeds=100)
    run_fullyear("virum")
    run_forecast("virum")
    run_baseline("virum")
    run_marketpower()
    run_pool_price()
    run_agent_table("virum")
    run_recycling("virum")
    run_reg_deployable("virum")
    if cp is not None:
        run_convex_audit("virum")
    print(f"\nAll experiments done in {time.time()-t0:.0f}s")

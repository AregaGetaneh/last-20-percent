"""Optimization models.

`solve_planner` is the centralized community mixed-integer quadratic program (SP);
with `relax=True` it returns the convex relaxation SP_rel. `solve_waterfall`
evaluates the ordered decentralized variants DE0, M1, IND, M2, M3, where M2 and M3
are price-taking peer-to-peer equilibria whose economic P2P and congestion prices
are recovered as weight-normalized duals. `price_discovery` is an illustrative
dual-decomposition routine and is not used for any reported value.

Prosumers are separately metered, but the coupling point carries only the net
community injection, so the hosting limit caps net export. Where the hosting
limit is not priced, a feasible DSO recourse curtails dispatched photovoltaic
power and reduces export by the same amount.
"""
from __future__ import annotations
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import config as C
from data import PilotData, build_pilot_data

DH_PRICE = 0.085                # district-heating energy price [EUR/kWh]
CARBON_PRICE = 0.10             # base carbon price [EUR/kg CO2]
EPS_SPILL = 1e-3                # tie-breaking penalty on renewable spill [EUR/kWh]
EPS_DUMP = 1e-4                 # tie-breaking penalty on heat dump [EUR/kWh]
EPS_REG = 1e-6                  # quadratic regularizer on grid and pool flows [EUR/(kW^2 h)]
REG = EPS_REG
ERR_PRIV = 0.25                 # private day-ahead forecast error (relative std)
ERR_SHARED = 0.08               # shared forecast error after information pooling


# --------------------------------------------------------------------------
# Centralized planner
# --------------------------------------------------------------------------
def solve_planner(d: PilotData, carbon_price: float = CARBON_PRICE,
                  dh_price: float = DH_PRICE, hosting: float | None = None,
                  import_cap: float | None = None,
                  relax: bool = False, verbose: bool = False,
                  regularization: float = EPS_REG) -> dict:
    T, dt = d.T, d.dt
    w = d.days_weight * dt
    A = list(d.agents.keys())
    Emax = d.hosting_limit if hosting is None else hosting
    SL = C.HOURS_PER_SEASON
    nseason = max(1, T // SL)

    m = gp.Model(f"SP_{d.pid}")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.Threads = C.THREADS
    m.Params.Seed = C.SEED
    m.Params.MIPGap = C.MIP_GAP
    m.Params.TimeLimit = 240

    ts = range(T)
    PV, chB, disB, eB, uB = {}, {}, {}, {}, {}
    chT, disT, eT = {}, {}, {}
    Phe, Pres, Qdump, DH, dsh, qsh, imp, exp, sell, buy = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for a in A:
        ag = d.agents[a]
        shcap = C.W_SHIFT * (float(np.sum(ag["Dshbase"])) / max(T, 1)) if ag["flex"] else 0.0
        big = 5.0 * (float(np.max(ag["Dfix"] + ag["Dshbase"])) + float(np.max(ag["PVavail"])) + ag["batt_kw"] + 1.0)
        for t in ts:
            PV[a, t] = m.addVar(ub=ag["PVavail"][t])
            bkw, bkwh = ag["batt_kw"], ag["batt_kwh"]
            chB[a, t] = m.addVar(ub=bkw); disB[a, t] = m.addVar(ub=bkw)
            uB[a, t] = m.addVar(vtype=(GRB.CONTINUOUS if relax else GRB.BINARY), lb=0, ub=1) if bkwh > 0 else None
            chT[a, t] = m.addVar(ub=ag["tes_kw"]); disT[a, t] = m.addVar(ub=ag["tes_kw"])
            Phe[a, t] = m.addVar(ub=(2e4 if ag["hp"] else 0.0))
            Pres[a, t] = m.addVar(ub=2e4)
            Qdump[a, t] = m.addVar()
            DH[a, t] = m.addVar(ub=(2e4 if ag["dh"] else 0.0))
            dsh[a, t] = m.addVar(lb=0.0, ub=(2.0 * ag["Dshbase"][t] + 1e-6) if ag["flex"] else ag["Dshbase"][t])
            imp[a, t] = m.addVar(ub=big); exp[a, t] = m.addVar(ub=big)
            sell[a, t] = m.addVar(ub=big); buy[a, t] = m.addVar(ub=big)
        for t in list(ts) + [T]:
            eB[a, t] = m.addVar(lb=C.SOC_MIN * ag["batt_kwh"], ub=C.SOC_MAX * ag["batt_kwh"]) if ag["batt_kwh"] > 0 else m.addVar(ub=0)
            eT[a, t] = m.addVar(ub=ag["tes_kwh"])
            qsh[a, t] = m.addVar(lb=-shcap, ub=shcap)
    Inet = {t: m.addVar() for t in ts} if import_cap is not None else None
    m.update()

    for a in A:
        ag = d.agents[a]
        for t in ts:
            m.addConstr(PV[a, t] + disB[a, t] + imp[a, t] + buy[a, t]
                        == ag["Dfix"][t] + dsh[a, t] + chB[a, t] + Phe[a, t] + Pres[a, t]
                        + exp[a, t] + sell[a, t], name=f"ebal_{a}_{t}")
            cop = float(np.clip(C.COP0 + C.KAPPA_COP * (d.Tamb[t] - 7.0), C.COP_MIN, C.COP_MAX))
            m.addConstr(cop * Phe[a, t] + Pres[a, t] + ag["Qpvt"][t] + disT[a, t] + DH[a, t]
                        == ag["Dheat"][t] + chT[a, t] + Qdump[a, t], name=f"hbal_{a}_{t}")
            m.addConstr(Qdump[a, t] <= ag["Qpvt"][t])
            if ag["batt_kwh"] > 0:
                m.addConstr(chB[a, t] <= ag["batt_kw"] * uB[a, t])
                m.addConstr(disB[a, t] <= ag["batt_kw"] * (1 - uB[a, t]))
            m.addConstr(eB[a, t + 1] == eB[a, t] + C.ETA_CH * chB[a, t] * dt - (disB[a, t] / C.ETA_DIS) * dt)
            m.addConstr(eT[a, t + 1] == (1 - C.KAPPA_TES * dt) * eT[a, t] + C.ETA_TES * chT[a, t] * dt - (disT[a, t] / C.ETA_TES) * dt)
            m.addConstr(qsh[a, t + 1] == qsh[a, t] + ag["Dshbase"][t] - dsh[a, t])
        for k in range(1, nseason + 1):
            m.addConstr(eB[a, SL * k] == eB[a, 0])
            m.addConstr(eT[a, SL * k] == eT[a, 0])
        for j in range(0, T + 1, 24):
            m.addConstr(qsh[a, j] == 0)
        if ag["flex"]:
            for t in ts:
                cap = C.ALPHA_SHIFT * (ag["Dfix"][t] + ag["Dshbase"][t])
                m.addConstr(dsh[a, t] - ag["Dshbase"][t] <= cap)
                m.addConstr(ag["Dshbase"][t] - dsh[a, t] <= cap)

    for t in ts:
        m.addConstr(gp.quicksum(sell[a, t] for a in A) == gp.quicksum(buy[a, t] for a in A))
        m.addConstr(gp.quicksum(exp[a, t] - imp[a, t] for a in A) <= Emax)
        if Inet is not None:
            m.addConstr(Inet[t] >= gp.quicksum(imp[a, t] - exp[a, t] for a in A))
    if import_cap is not None:
        m.addConstr(gp.quicksum(w[t] * Inet[t] for t in ts) <= import_cap)

    econ = gp.quicksum(w[t] * gp.quicksum(
        d.price_imp[t] * imp[a, t] - d.price_exp[t] * exp[a, t]
        + d.tariff[t] * (imp[a, t] + exp[a, t])
        + C.DEG_COST * (chB[a, t] + disB[a, t]) + dh_price * DH[a, t] for a in A) for t in ts)
    co2 = gp.quicksum(w[t] * gp.quicksum(d.ef[t] * imp[a, t] + d.ef_dh * DH[a, t] for a in A) for t in ts)
    tie = gp.quicksum(w[t] * gp.quicksum(
        EPS_SPILL * (d.agents[a]["PVavail"][t] - PV[a, t]) + EPS_DUMP * Qdump[a, t] for a in A) for t in ts)
    reg = gp.quicksum(w[t] * gp.quicksum(
        regularization * (imp[a, t] * imp[a, t] + exp[a, t] * exp[a, t]
                          + buy[a, t] * buy[a, t] + sell[a, t] * sell[a, t]) for a in A) for t in ts)
    m.setObjective(econ + carbon_price * co2 + tie + reg, GRB.MINIMIZE)

    m.optimize()
    if m.SolCount == 0:
        return {"pid": d.pid, "status": int(m.Status), "ok": False}
    r = _extract(d, PV, chB, disB, DH, Phe, Pres, dsh, imp, exp, w, dh_price, carbon_price, Emax)
    r["obj"] = float(m.ObjVal)
    r["simul_kWh"] = sum(float(np.sum(w * np.minimum(
        np.array([chB[a, t].X for t in ts]), np.array([disB[a, t].X for t in ts])))) for a in A) / 1000
    pv_disp = np.array([sum(PV[a, t].X for a in A) for t in ts])
    E_net = np.maximum(np.array([sum(exp[a, t].X - imp[a, t].X for a in A) for t in ts]), 0.0)
    r["export_pv_cov"] = float(np.sum(np.minimum(E_net, pv_disp) * w)) / max(float(np.sum(E_net * w)), 1e-9)
    m.dispose()
    return r


def _extract(d, PV, chB, disB, DH, Phe, Pres, dsh, imp, exp, w, dh_price, carbon_price, Emax):
    T = d.T
    A = list(d.agents.keys())
    g = lambda V: np.array([sum(V[a, t].X for a in A) for t in range(T)])
    impv, expv = g(imp), g(exp)
    net = impv - expv
    I_net = np.maximum(net, 0.0)
    E_net = np.maximum(-net, 0.0)
    pv_disp = g(PV)
    pv_avail = sum(d.agents[a]["PVavail"] for a in A)
    spill = pv_avail - pv_disp
    dem = np.zeros(T)
    for t in range(T):
        dem[t] = sum(d.agents[a]["Dfix"][t] + dsh[a, t].X + chB[a, t].X + Phe[a, t].X + Pres[a, t].X
                     - disB[a, t].X for a in A)
    dhv = g(DH)
    tot = lambda x: float(np.sum(x * w)) / 1000
    imp_g, exp_g = tot(impv), tot(expv)
    inet_MWh, enet_MWh = tot(I_net), tot(E_net)
    spill_MWh = tot(spill)
    dem_MWh, pv_MWh, dh_MWh = tot(dem), tot(pv_avail), tot(dhv)
    c_imp = float(np.sum(w * d.price_imp * impv)) / 1000
    c_exprev = -float(np.sum(w * d.price_exp * expv)) / 1000
    c_tariff = float(np.sum(w * d.tariff * (impv + expv))) / 1000
    c_deg = sum(float(np.sum(w * C.DEG_COST * np.array([chB[a, t].X + disB[a, t].X for t in range(T)]))) for a in A) / 1000
    c_dh = dh_price * dh_MWh
    co2_t = float(np.sum(w * d.ef * impv)) / 1000 + d.ef_dh * dh_MWh
    c_carbon = carbon_price * co2_t
    cost_op = c_imp + c_exprev + c_tariff + c_deg + c_dh              # private operating cost
    cost_kEUR = cost_op + c_carbon                                   # reported economic cost
    ssr = 1 - inet_MWh / max(dem_MWh, 1e-9)               # self-sufficiency ratio (SSR): demand met without net import
    lpi_os = (enet_MWh + spill_MWh) / max(pv_MWh, 1e-9)   # oversupply loss-of-positivity index (LPI_os): renewable exported or curtailed
    match_rate = float(np.sum((I_net <= 1e-6) * w)) / max(float(np.sum(w)), 1e-9)
    return dict(
        pid=d.pid, status="optimal", ok=True,
        pv_MWh=pv_MWh, dem_MWh=dem_MWh,
        imp_MWh=inet_MWh, exp_MWh=enet_MWh, curt_MWh=spill_MWh,
        imp_gross_MWh=imp_g, exp_gross_MWh=exp_g, dh_MWh=dh_MWh,
        cost_kEUR=cost_kEUR, cost_op_kEUR=cost_op, carbon_kEUR=c_carbon,
        co2_t=co2_t, SSR=ssr, LPI_os=lpi_os, match_rate=match_rate,
        cost_parts=dict(purchase=c_imp, export_rev=c_exprev, tariff=c_tariff, degradation=c_deg,
                        district_heat=c_dh, carbon=c_carbon),
        series=dict(I=I_net, E=E_net, CURT=spill, PV=pv_avail, PVdisp=pv_disp, dem=dem,
                    imp_gross=impv, exp_gross=expv),
    )


# --------------------------------------------------------------------------
# Decentralized variants and the coordination waterfall
# --------------------------------------------------------------------------
def _flow_bound(ag: dict) -> float:
    return 5.0 * (float(np.max(ag["Dfix"] + ag["Dshbase"]))
                  + float(np.max(ag["PVavail"])) + float(ag["batt_kw"]) + 1.0)


def _agent_lp(d, aid, steps, pvav, dfix, dshb, dheat, e0, cyclic_season,
              carbon_price, dh_price, fix=None, regularization: float = REG, terminal=None):
    """Single prosumer problem over `steps`. In rolling operation the committed
    battery charge/discharge is fixed in the realization pass through `fix`; grid
    exchange, dispatch, and the within-day shiftable load remain real-time recourse.
    `terminal` pins the end-of-window battery state."""
    ag = d.agents[aid]; dt = d.dt
    n = len(steps)
    shcap = C.W_SHIFT * (float(np.sum(dshb)) / max(n, 1)) if ag["flex"] else 0.0
    big = _flow_bound(ag)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 1
    PV = m.addVars(n, ub=[pvav[k] for k in range(n)])
    chB = m.addVars(n, ub=ag["batt_kw"]); disB = m.addVars(n, ub=ag["batt_kw"])
    chT = m.addVars(n, ub=ag["tes_kw"]); disT = m.addVars(n, ub=ag["tes_kw"])
    Phe = m.addVars(n, ub=(2e4 if ag["hp"] else 0.0)); Pres = m.addVars(n, ub=2e4)
    Qd = m.addVars(n); DH = m.addVars(n, ub=(2e4 if ag["dh"] else 0.0))
    dsh = m.addVars(n, lb=0.0, ub=[(2 * dshb[k] + 1e-6) if ag["flex"] else dshb[k] for k in range(n)])
    imp = m.addVars(n, ub=big); exp = m.addVars(n, ub=big)
    eB = m.addVars(n + 1, lb=C.SOC_MIN * ag["batt_kwh"], ub=max(C.SOC_MAX * ag["batt_kwh"], 1e-6))
    eT = m.addVars(n + 1, ub=max(ag["tes_kwh"], 1e-6))
    qsh = m.addVars(n + 1, lb=-shcap, ub=shcap)
    for k in range(n):
        t = steps[k]
        cop = float(np.clip(C.COP0 + C.KAPPA_COP * (d.Tamb[t] - 7.0), C.COP_MIN, C.COP_MAX))
        m.addConstr(PV[k] + disB[k] + imp[k] == dfix[k] + dsh[k] + chB[k] + Phe[k] + Pres[k] + exp[k])
        m.addConstr(cop * Phe[k] + Pres[k] + ag["Qpvt"][t] + disT[k] + DH[k] == dheat[k] + chT[k] + Qd[k])
        m.addConstr(Qd[k] <= ag["Qpvt"][t])
        m.addConstr(eB[k + 1] == eB[k] + C.ETA_CH * chB[k] * dt - (disB[k] / C.ETA_DIS) * dt)
        m.addConstr(eT[k + 1] == (1 - C.KAPPA_TES * dt) * eT[k] + C.ETA_TES * chT[k] * dt - (disT[k] / C.ETA_TES) * dt)
        m.addConstr(qsh[k + 1] == qsh[k] + dshb[k] - dsh[k])
        if ag["flex"]:
            cap = C.ALPHA_SHIFT * (dfix[k] + dshb[k])
            m.addConstr(dsh[k] - dshb[k] <= cap); m.addConstr(dshb[k] - dsh[k] <= cap)
    if cyclic_season:
        SL = C.HOURS_PER_SEASON
        for kb in range(0, n + 1, SL):
            m.addConstr(eB[kb] == eB[0]); m.addConstr(eT[kb] == eT[0])
    else:
        m.addConstr(eB[0] == e0.get("eB", 0.5 * ag["batt_kwh"]))
        m.addConstr(eT[0] == e0.get("eT", 0.5 * ag["tes_kwh"]))
    m.addConstr(qsh[0] == 0)
    for j in range(0, n + 1, 24):
        m.addConstr(qsh[j] == 0)
    if fix is not None:
        for k in range(n):
            m.addConstr(chB[k] == fix["chB"][k]); m.addConstr(disB[k] == fix["disB"][k])
    if terminal is not None:
        if "eB" in terminal:
            m.addConstr(eB[n] == terminal["eB"])
        if "eT" in terminal:
            m.addConstr(eT[n] == terminal["eT"])
    w = d.days_weight * d.dt
    obj = gp.QuadExpr()
    for k in range(n):
        t = steps[k]
        obj += w[t] * (d.price_imp[t] * imp[k] - d.price_exp[t] * exp[k]
                       + d.tariff[t] * (imp[k] + exp[k])
                       + C.DEG_COST * (chB[k] + disB[k]) + dh_price * DH[k]
                       + carbon_price * (d.ef[t] * imp[k] + d.ef_dh * DH[k])
                       + EPS_SPILL * (pvav[k] - PV[k]) + EPS_DUMP * Qd[k]
                       + regularization * (imp[k] * imp[k] + exp[k] * exp[k]))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        m.dispose()
        raise RuntimeError(f"Agent problem failed for {aid}: status={m.Status}")
    G = lambda v: np.array([v[k].X for k in range(n)])
    out = dict(imp=G(imp), exp=G(exp), PV=G(PV), chB=G(chB), disB=G(disB), chT=G(chT), disT=G(disT),
               DH=G(DH), Phe=G(Phe), Pres=G(Pres), dsh=G(dsh), Qd=G(Qd),
               eB_end=eB[n].X, eT_end=eT[n].X)
    m.dispose()
    return out


def _independent(d, mode, carbon_price, dh_price, seed=0, regularization: float = REG,
                 horizon: int = 24):
    """Independent (no-pool) variants.
      mode='de0' : private daily-rolling forecast (plan on forecast, realize on truth).
      mode='m1'  : shared daily-rolling forecast (better information, same horizon).
      mode='ind' : perfect-information, full-horizon control with season-cyclic storage."""
    A = list(d.agents.keys()); T = d.T
    rng = np.random.default_rng(C.SEED + seed)
    keys = ["imp", "exp", "PV", "chB", "disB", "DH", "Phe", "Pres", "dsh"]
    agg = {k: np.zeros(T) for k in keys}
    if mode == "ind":
        steps = list(range(T))
        for a in A:
            ag = d.agents[a]
            r = _agent_lp(d, a, steps, ag["PVavail"], ag["Dfix"], ag["Dshbase"], ag["Dheat"],
                          {}, True, carbon_price, dh_price, regularization=regularization)
            for k in keys:
                agg[k] += r[k]
        return agg, None, None
    err = ERR_PRIV if mode == "de0" else ERR_SHARED
    windows = list(range(0, T, horizon))
    nwin = len(windows)
    # Day-varying forecast: fresh draw per daily window; shared across agents for M1,
    # private per agent for DE0.
    common_fe = None
    if mode == "m1":
        common_fe = [{q: np.clip(1 + rng.normal(0, err, min(d0 + horizon, T) - d0), 0.2, None)
                      for q in ["pv", "load", "heat"]} for d0 in windows]
    SL = C.HOURS_PER_SEASON
    for a in A:
        ag = d.agents[a]
        ref = {"eB": 0.5 * ag["batt_kwh"], "eT": 0.5 * ag["tes_kwh"]}
        e0 = dict(ref)
        for wi, d0 in enumerate(windows):
            steps = list(range(d0, min(d0 + horizon, T)))
            end = min(d0 + horizon, T); sl = slice(d0, end); L = len(steps)
            # Reset storage to the reference state at each season boundary and close it
            # there; state is carried only within a representative week.
            if d0 % SL == 0:
                e0 = dict(ref)
            pv_t = ag["PVavail"][sl]; df_t = ag["Dfix"][sl]; ds_t = ag["Dshbase"][sl]; dh_t = ag["Dheat"][sl]
            fe = common_fe[wi] if mode == "m1" else {q: np.clip(1 + rng.normal(0, err, L), 0.2, None)
                                                     for q in ["pv", "load", "heat"]}
            last_of_week = (end % SL == 0 or wi == nwin - 1)
            terminal = dict(ref) if last_of_week else None
            # plan on the day-ahead forecast, committing the battery schedule
            plan = _agent_lp(d, a, steps, pv_t * fe["pv"], df_t * fe["load"],
                             ds_t * fe["load"], dh_t * fe["heat"], e0, False, carbon_price, dh_price,
                             regularization=regularization, terminal=terminal)
            # Realize on truth with the battery schedule fixed; thermal, grid, and
            # shiftable load are recourse, with the thermal terminal re-imposed at week end.
            close = {"eT": ref["eT"]} if last_of_week else None
            real = _agent_lp(d, a, steps, pv_t, df_t, ds_t, dh_t, e0, False, carbon_price, dh_price,
                             fix=plan, regularization=regularization, terminal=close)
            for k in keys:
                agg[k][sl] += real[k]
            e0 = {"eB": real["eB_end"], "eT": real["eT_end"]}
    return agg, None, None


def _joint(d, grid_aware, carbon_price, dh_price, detail=False,
           pool_enabled: bool = True, regularization: float = REG):
    """Perfect-information community problem. `grid_aware=True` prices the hosting
    constraint (M3); the economic prices are the weight-normalized duals."""
    A = list(d.agents.keys()); T = d.T; dt = d.dt
    SL = C.HOURS_PER_SEASON; nseason = max(1, T // SL)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 1
    V = {}
    for a in A:
        ag = d.agents[a]
        shcap = C.W_SHIFT * (float(np.sum(ag["Dshbase"])) / max(T, 1)) if ag["flex"] else 0.0
        big = _flow_bound(ag)
        pool_ub = big if pool_enabled else 0.0
        V[a] = dict(
            PV=m.addVars(T, ub=[ag["PVavail"][t] for t in range(T)]),
            chB=m.addVars(T, ub=ag["batt_kw"]), disB=m.addVars(T, ub=ag["batt_kw"]),
            chT=m.addVars(T, ub=ag["tes_kw"]), disT=m.addVars(T, ub=ag["tes_kw"]),
            Phe=m.addVars(T, ub=(2e4 if ag["hp"] else 0.0)), Pres=m.addVars(T, ub=2e4),
            Qd=m.addVars(T), DH=m.addVars(T, ub=(2e4 if ag["dh"] else 0.0)),
            dsh=m.addVars(T, lb=0.0, ub=[(2 * ag["Dshbase"][t] + 1e-6) if ag["flex"] else ag["Dshbase"][t] for t in range(T)]),
            imp=m.addVars(T, ub=big), exp=m.addVars(T, ub=big),
            buy=m.addVars(T, ub=pool_ub), sell=m.addVars(T, ub=pool_ub),
            eB=m.addVars(T + 1, lb=C.SOC_MIN * ag["batt_kwh"], ub=max(C.SOC_MAX * ag["batt_kwh"], 1e-6)),
            eT=m.addVars(T + 1, ub=max(ag["tes_kwh"], 1e-6)),
            qsh=m.addVars(T + 1, lb=-shcap, ub=shcap),
        )
    poolcon, hostcon = {}, {}
    for a in A:
        ag = d.agents[a]; v = V[a]
        for t in range(T):
            cop = float(np.clip(C.COP0 + C.KAPPA_COP * (d.Tamb[t] - 7.0), C.COP_MIN, C.COP_MAX))
            m.addConstr(v["PV"][t] + v["disB"][t] + v["imp"][t] + v["buy"][t]
                        == ag["Dfix"][t] + v["dsh"][t] + v["chB"][t] + v["Phe"][t] + v["Pres"][t] + v["exp"][t] + v["sell"][t])
            m.addConstr(cop * v["Phe"][t] + v["Pres"][t] + ag["Qpvt"][t] + v["disT"][t] + v["DH"][t]
                        == ag["Dheat"][t] + v["chT"][t] + v["Qd"][t])
            m.addConstr(v["Qd"][t] <= ag["Qpvt"][t])
            m.addConstr(v["eB"][t + 1] == v["eB"][t] + C.ETA_CH * v["chB"][t] * dt - (v["disB"][t] / C.ETA_DIS) * dt)
            m.addConstr(v["eT"][t + 1] == (1 - C.KAPPA_TES * dt) * v["eT"][t] + C.ETA_TES * v["chT"][t] * dt - (v["disT"][t] / C.ETA_TES) * dt)
            m.addConstr(v["qsh"][t + 1] == v["qsh"][t] + ag["Dshbase"][t] - v["dsh"][t])
            if ag["flex"]:
                cap = C.ALPHA_SHIFT * (ag["Dfix"][t] + ag["Dshbase"][t])
                m.addConstr(v["dsh"][t] - ag["Dshbase"][t] <= cap); m.addConstr(ag["Dshbase"][t] - v["dsh"][t] <= cap)
        for k in range(1, nseason + 1):
            m.addConstr(v["eB"][SL * k] == v["eB"][0]); m.addConstr(v["eT"][SL * k] == v["eT"][0])
        for j in range(0, T + 1, 24):
            m.addConstr(v["qsh"][j] == 0)
    for t in range(T):
        if pool_enabled:
            poolcon[t] = m.addConstr(gp.quicksum(V[a]["sell"][t] - V[a]["buy"][t] for a in A) == 0)
        if grid_aware:
            hostcon[t] = m.addConstr(gp.quicksum(V[a]["exp"][t] - V[a]["imp"][t] for a in A) <= d.hosting_limit)
    w = d.days_weight * d.dt
    cost = gp.quicksum(w[t] * gp.quicksum(
        d.price_imp[t] * V[a]["imp"][t] - d.price_exp[t] * V[a]["exp"][t]
        + d.tariff[t] * (V[a]["imp"][t] + V[a]["exp"][t])
        + C.DEG_COST * (V[a]["chB"][t] + V[a]["disB"][t]) + dh_price * V[a]["DH"][t]
        + carbon_price * (d.ef[t] * V[a]["imp"][t] + d.ef_dh * V[a]["DH"][t])
        + EPS_SPILL * (ag_pvav(d, a, t) - V[a]["PV"][t]) + EPS_DUMP * V[a]["Qd"][t]
        + regularization * (V[a]["imp"][t] * V[a]["imp"][t] + V[a]["exp"][t] * V[a]["exp"][t]
                            + V[a]["buy"][t] * V[a]["buy"][t] + V[a]["sell"][t] * V[a]["sell"][t])
        for a in A) for t in range(T))
    m.setObjective(cost, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        m.dispose()
        raise RuntimeError(f"Joint problem failed: status={m.Status}")
    keys = ["imp", "exp", "PV", "chB", "disB", "DH", "Phe", "Pres", "dsh", "buy", "sell"]
    agg = {k: np.zeros(T) for k in keys}
    for a in A:
        for k in keys:
            agg[k] += np.array([V[a][k][t].X for t in range(T)])
    pi = np.array([poolcon[t].Pi / max(w[t], 1e-12) for t in range(T)]) if pool_enabled else np.zeros(T)
    tau = np.array([-hostcon[t].Pi / max(w[t], 1e-12) for t in range(T)]) if grid_aware else np.zeros(T)
    per_agent = None
    if detail:
        per_agent = {}
        for a in A:
            v = V[a]; g = lambda k: np.array([v[k][t].X for t in range(T)])
            imp_a, exp_a, buy_a, sell_a = g("imp"), g("exp"), g("buy"), g("sell")
            dh_a = g("DH"); chb, disb = g("chB"), g("disB")
            bill = float(np.sum(w * (d.price_imp * imp_a - d.price_exp * exp_a
                        + d.tariff * (imp_a + exp_a) + C.DEG_COST * (chb + disb) + dh_price * dh_a
                        + carbon_price * (d.ef * imp_a + d.ef_dh * dh_a)
                        + pi * (buy_a - sell_a) + tau * (exp_a - imp_a)))) / 1000
            per_agent[a] = dict(bill=bill, imp=imp_a, exp=exp_a, buy=buy_a, sell=sell_a)
    m.dispose()
    return (agg, pi, tau, per_agent) if detail else (agg, pi, tau)


def ag_pvav(d, a, t):
    return d.agents[a]["PVavail"][t]


def _kpis(d, agg, name, real_hosting, carbon_price=CARBON_PRICE):
    """Net-based KPIs. Where the hosting limit is not priced, the DSO recourse
    throttles net export to the limit by curtailing dispatched PV."""
    T = d.T; w = d.days_weight * d.dt
    A = list(d.agents.keys())
    impv, expv = agg["imp"], agg["exp"]
    net = impv - expv
    I_net = np.maximum(net, 0.0); E_net_desired = np.maximum(-net, 0.0)
    excess = np.maximum(E_net_desired - real_hosting, 0.0)
    curt_host = np.minimum(excess, agg["PV"])
    E_net = E_net_desired - curt_host
    residual = float(np.sum((excess - curt_host) * w)) / 1000
    pv_avail = sum(d.agents[a]["PVavail"] for a in A)
    spill = pv_avail - agg["PV"] + curt_host
    dem = np.zeros(T)
    for a in A:
        dem = dem + d.agents[a]["Dfix"]
    dem = dem + agg["dsh"] + agg["Phe"] + agg["Pres"] + agg["chB"] - agg["disB"]
    tot = lambda x: float(np.sum(x * w)) / 1000
    inet, enet, sp = tot(I_net), tot(E_net), tot(spill)
    dem_MWh, pv_MWh, dh_MWh = tot(dem), tot(pv_avail), tot(agg["DH"])
    exp_eff = np.maximum(expv - curt_host, 0.0)
    cost_op = float(np.sum(w * (d.price_imp * impv - d.price_exp * exp_eff + d.tariff * (impv + exp_eff)))) / 1000 \
        + float(np.sum(w * C.DEG_COST * (agg["chB"] + agg["disB"]))) / 1000 + DH_PRICE * dh_MWh
    co2_t = float(np.sum(w * d.ef * impv)) / 1000 + d.ef_dh * dh_MWh
    carbon = carbon_price * co2_t
    cost = cost_op + carbon
    # self-sufficiency ratio (SSR): share of served demand met without net grid import
    ssr = 1 - inet / max(dem_MWh, 1e-9)
    # oversupply loss-of-positivity index (LPI_os): share of available renewable
    # generation exported or curtailed rather than self-consumed
    lpi_os = (enet + sp) / max(pv_MWh, 1e-9)
    # hourly matching rate (HMR): duration-weighted share of hours with no net import
    match = float(np.sum((I_net <= 1e-6) * w)) / max(float(np.sum(w)), 1e-9)
    return dict(variant=name, pid=d.pid, cost_kEUR=cost, cost_op_kEUR=cost_op, carbon_kEUR=carbon,
                co2_t=co2_t, SSR=ssr, LPI_os=lpi_os, match_rate=match,
                imp_MWh=inet, exp_MWh=enet, curt_MWh=sp, host_residual_MWh=residual)


def solve_waterfall(d: PilotData, carbon_price=CARBON_PRICE, dh_price=DH_PRICE,
                    regularization: float = REG) -> tuple[dict, dict]:
    H = d.hosting_limit
    out, prices = {}, {}
    a0, _, _ = _independent(d, "de0", carbon_price, dh_price, regularization=regularization)
    out["DE0"] = _kpis(d, a0, "DE0", H, carbon_price)
    a1, _, _ = _independent(d, "m1", carbon_price, dh_price, regularization=regularization)
    out["DE_M1"] = _kpis(d, a1, "DE_M1", H, carbon_price)
    ai, _, _ = _independent(d, "ind", carbon_price, dh_price, regularization=regularization)
    out["DE_IND"] = _kpis(d, ai, "DE_IND", H, carbon_price)
    a2, pi2, tau2 = _joint(d, False, carbon_price, dh_price, regularization=regularization)
    out["DE_M2"] = _kpis(d, a2, "DE_M2", H, carbon_price); prices["DE_M2"] = (pi2, tau2)
    a3, pi3, tau3 = _joint(d, True, carbon_price, dh_price, regularization=regularization)
    out["DE_M3"] = _kpis(d, a3, "DE_M3", H, carbon_price); prices["DE_M3"] = (pi3, tau3)
    return out, prices


def price_discovery(d, carbon_price=CARBON_PRICE, dh_price=DH_PRICE,
                    iters=300, gamma0=0.04, regularization: float = REG):
    """Illustrative dual-decomposition price discovery on the raw multipliers of
    the pool-balance and net-export constraints. Diagnostic only."""
    A = list(d.agents.keys()); T = d.T; w = d.days_weight * d.dt
    steps = list(range(T)); Emax = d.hosting_limit
    flow_scale = max(1.0, float(np.mean([np.max(d.agents[a]["PVavail"]) for a in A])) * len(A))
    cost_scale = max(1.0, float(np.sum(w * d.price_imp)) * flow_scale)
    pi0 = 0.5 * ((d.price_exp - d.tariff) + (d.price_imp + d.tariff + carbon_price * d.ef))
    lam_pool = -w * pi0
    lam_host = np.zeros(T)
    sum_gamma = 0.0
    pi_sum = np.zeros(T); tau_sum = np.zeros(T)
    xsum = {k: np.zeros(T) for k in ["net_export", "pool_imbalance"]}
    hist = {"pool_residual": [], "hosting_residual": [], "complementarity": [], "step": []}
    best = None; best_res = np.inf
    for k in range(iters):
        gamma = gamma0 / np.sqrt(k + 1.0)
        pi = -lam_pool / np.maximum(w, 1e-12)
        tau = lam_host / np.maximum(w, 1e-12)
        sell_tot = np.zeros(T); buy_tot = np.zeros(T)
        exp_tot = np.zeros(T); imp_tot = np.zeros(T)
        for a in A:
            r = _agent_lp_priced(d, a, steps, carbon_price, dh_price, pi, tau,
                                 regularization=regularization)
            sell_tot += r["sell"]; buy_tot += r["buy"]
            exp_tot += r["exp"]; imp_tot += r["imp"]
        g_pool = sell_tot - buy_tot
        g_host = (exp_tot - imp_tot) - Emax
        rp_pool = float(np.linalg.norm(g_pool) / (flow_scale * np.sqrt(T)))
        rp_host = float(np.linalg.norm(np.maximum(g_host, 0.0)) / (flow_scale * np.sqrt(T)))
        rc = float(np.sum(w * tau * np.maximum(-g_host, 0.0)) / cost_scale)
        score = rp_pool + rp_host + rc
        if score < best_res:
            best_res = score
            best = dict(pi=pi.copy(), tau=tau.copy(), net_export=(exp_tot - imp_tot).copy(),
                        pool_imbalance=(buy_tot - sell_tot).copy())
        sum_gamma += gamma
        pi_sum += gamma * pi; tau_sum += gamma * tau
        xsum["net_export"] += gamma * (exp_tot - imp_tot)
        xsum["pool_imbalance"] += gamma * (buy_tot - sell_tot)
        hist["pool_residual"].append(rp_pool)
        hist["hosting_residual"].append(rp_host)
        hist["complementarity"].append(rc)
        hist["step"].append(gamma)
        lam_pool = lam_pool + gamma * (g_pool / flow_scale)
        lam_host = np.maximum(0.0, lam_host + gamma * (g_host / flow_scale))
    return dict(
        hist=hist,
        pi=pi_sum / max(sum_gamma, 1e-12),
        tau=tau_sum / max(sum_gamma, 1e-12),
        coupling_average={k: v / max(sum_gamma, 1e-12) for k, v in xsum.items()},
        lowest_residual=best,
        normalization=dict(flow_scale=flow_scale, cost_scale=cost_scale),
    )


def _agent_lp_priced(d, aid, steps, carbon_price, dh_price, pi, tau,
                     regularization: float = REG):
    """Full-horizon agent best response to a pool price pi and congestion price tau."""
    ag = d.agents[aid]; dt = d.dt; n = len(steps)
    shcap = C.W_SHIFT * (float(np.sum(ag["Dshbase"])) / max(n, 1)) if ag["flex"] else 0.0
    SL = C.HOURS_PER_SEASON
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 1
    PV = m.addVars(n, ub=[ag["PVavail"][steps[k]] for k in range(n)])
    chB = m.addVars(n, ub=ag["batt_kw"]); disB = m.addVars(n, ub=ag["batt_kw"])
    chT = m.addVars(n, ub=ag["tes_kw"]); disT = m.addVars(n, ub=ag["tes_kw"])
    Phe = m.addVars(n, ub=(2e4 if ag["hp"] else 0.0)); Pres = m.addVars(n, ub=2e4)
    Qd = m.addVars(n); DH = m.addVars(n, ub=(2e4 if ag["dh"] else 0.0))
    dsh = m.addVars(n, lb=0.0, ub=[(2 * ag["Dshbase"][steps[k]] + 1e-6) if ag["flex"] else ag["Dshbase"][steps[k]] for k in range(n)])
    big = _flow_bound(ag)
    imp = m.addVars(n, ub=big); exp = m.addVars(n, ub=big)
    buy = m.addVars(n, ub=big); sell = m.addVars(n, ub=big)
    eB = m.addVars(n + 1, lb=C.SOC_MIN * ag["batt_kwh"], ub=max(C.SOC_MAX * ag["batt_kwh"], 1e-6))
    eT = m.addVars(n + 1, ub=max(ag["tes_kwh"], 1e-6))
    qsh = m.addVars(n + 1, lb=-shcap, ub=shcap)
    for k in range(n):
        t = steps[k]
        cop = float(np.clip(C.COP0 + C.KAPPA_COP * (d.Tamb[t] - 7.0), C.COP_MIN, C.COP_MAX))
        m.addConstr(PV[k] + disB[k] + imp[k] + buy[k] == ag["Dfix"][t] + dsh[k] + chB[k] + Phe[k] + Pres[k] + exp[k] + sell[k])
        m.addConstr(cop * Phe[k] + Pres[k] + ag["Qpvt"][t] + disT[k] + DH[k] == ag["Dheat"][t] + chT[k] + Qd[k])
        m.addConstr(Qd[k] <= ag["Qpvt"][t])
        m.addConstr(eB[k + 1] == eB[k] + C.ETA_CH * chB[k] * dt - (disB[k] / C.ETA_DIS) * dt)
        m.addConstr(eT[k + 1] == (1 - C.KAPPA_TES * dt) * eT[k] + C.ETA_TES * chT[k] * dt - (disT[k] / C.ETA_TES) * dt)
        m.addConstr(qsh[k + 1] == qsh[k] + ag["Dshbase"][t] - dsh[k])
        if ag["flex"]:
            cap = C.ALPHA_SHIFT * (ag["Dfix"][t] + ag["Dshbase"][t])
            m.addConstr(dsh[k] - ag["Dshbase"][t] <= cap); m.addConstr(ag["Dshbase"][t] - dsh[k] <= cap)
    for kb in range(0, n + 1, SL):
        m.addConstr(eB[kb] == eB[0]); m.addConstr(eT[kb] == eT[0])
    for j in range(0, n + 1, 24):
        m.addConstr(qsh[j] == 0)
    w = d.days_weight * d.dt
    obj = gp.QuadExpr()
    for k in range(n):
        t = steps[k]
        obj += w[t] * (d.price_imp[t] * imp[k] - d.price_exp[t] * exp[k] + d.tariff[t] * (imp[k] + exp[k])
                       + C.DEG_COST * (chB[k] + disB[k]) + dh_price * DH[k]
                       + carbon_price * (d.ef[t] * imp[k] + d.ef_dh * DH[k])
                       + EPS_SPILL * (ag["PVavail"][t] - PV[k]) + EPS_DUMP * Qd[k]
                       + pi[t] * (buy[k] - sell[k]) + tau[t] * (exp[k] - imp[k])
                       + regularization * (imp[k] * imp[k] + exp[k] * exp[k] + buy[k] * buy[k] + sell[k] * sell[k]))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        m.dispose()
        z = np.zeros(n)
        return dict(imp=z, exp=z, buy=z, sell=z)
    G = lambda v: np.array([v[k].X for k in range(n)])
    out = dict(imp=G(imp), exp=G(exp), buy=G(buy), sell=G(sell))
    m.dispose()
    return out


if __name__ == "__main__":
    d = build_pilot_data("virum")
    wf, pr = solve_waterfall(d)
    sp = solve_planner(d, relax=False)
    print(f"{'variant':8s} {'cost[k]':>8s} {'SSR':>6s} {'LPIos':>6s} {'curt':>7s}")
    for v in ["DE0", "DE_M1", "DE_IND", "DE_M2", "DE_M3"]:
        r = wf[v]
        print(f"{v:8s} {r['cost_kEUR']:8.1f} {r['SSR']:6.2f} {r['LPI_os']:6.2f} {r['curt_MWh']:7.1f}")
    print(f"{'SP':8s} {sp['cost_kEUR']:8.1f} {sp['SSR']:6.2f} {sp['LPI_os']:6.2f} {sp['curt_MWh']:7.1f}")

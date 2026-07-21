"""Regenerate the manuscript tables (LaTeX fragments) and figures (PDF/PNG) from
the cached results in results/*.json. Run `python outputs.py` after experiments.py.
Tables are written to tables/, figures to figures/.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from data import build_pilot_data, annual_summary

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_DIR, "results")
TABLES = os.path.join(_DIR, "tables")
FIGURES = os.path.join(_DIR, "figures")
GEO = os.path.join(_DIR, "data")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

ORDER = ["virum", "limerick", "vasteras", "graz", "turkey", "romania"]
DEEP = ["virum", "limerick", "vasteras", "graz"]
TEX_NAME = {"virum": "Virum", "limerick": "Limerick", "vasteras": "V\\\"aster\\aa s",
            "graz": "Graz--Berlin", "turkey": "Turkey", "romania": "Romania"}
FIG_NAME = {"virum": "Virum (DK)", "limerick": "Limerick (IE)", "vasteras": "Västerås (SE)",
            "graz": "Graz–Berlin (DE-AT)", "turkey": "Turkey", "romania": "Romania"}
VARS = ["DE0", "DE_M1", "DE_IND", "DE_M2", "DE_M3"]
VLAB = ["DE0", "+Info", "+Diag", "+Pool", "+Grid"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "savefig.bbox": "tight",
})


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _write_tex(name, body):
    with open(os.path.join(TABLES, name), "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  {name}")


def _save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES, name + ".pdf"))
    fig.savefig(os.path.join(FIGURES, name + ".png"))
    plt.close(fig)
    print(f"  {name}")


def _annual_weights(T):
    hps = C.HOURS_PER_SEASON
    w = np.concatenate([[C.SEASON_WEIGHT[s] / (hps / 24.0)] * hps for s in C.SEASONS]) * C.DT
    return w[:T]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def t_pilots(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; s = annual_summary(build_pilot_data(pid))
        rows.append(f"{TEX_NAME[pid]} & {r['country']} & {r['zone']} & {r['n_agents']} & "
                    f"{r['pv_kwp']:.0f} & {s['pv_MWh']:.0f} & {s['elec_dem_MWh']:.0f} & "
                    f"{s['heat_dem_MWh']:.0f} & {r['hosting']:.0f} & {r['tier']} \\\\")
    _write_tex("tab_pilots.tex",
               "\\begin{tabular}{llccrrrrrl}\n\\toprule\n"
               "Configuration & Country & Zone & Agents & PV & RES & Electricity\\ & Heat & Hosting & Tier \\\\\n"
               " & & & & [kWp] & [MWh] & [MWh] & [MWh] & [kW] & \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_main(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; sp = r["SP"]; d0 = r["variants"]["DE0"]
        gap = d0["cost_kEUR"] - sp["cost_kEUR"]
        rows.append(f"{TEX_NAME[pid]} & {sp['cost_kEUR']:.1f} & {sp['co2_t']:.1f} & "
                    f"{sp['SSR']:.2f} & {sp.get('match_rate', 0):.2f} & {sp['LPI_os']:.2f} & "
                    f"{sp['imp_MWh']:.0f} & {sp['curt_MWh']:.0f} & {gap:.2f} & "
                    f"{100*gap/sp['cost_kEUR']:.1f}\\% \\\\")
    _write_tex("tab_main.tex",
               "\\begin{tabular}{lrrcccrrrr}\n\\toprule\n"
               "District & Cost & CO$_2$ & SSR & HMR & LPI$_{os}$ & Import & Curtail & Gap & Gap \\\\\n"
               " & [k\\euro] & [t] & & & & [MWh] & [MWh] & [k\\euro] & \\% \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_waterfall(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; v = r["variants"]; sp = r["SP"]
        c = {k: v[k]["cost_kEUR"] for k in VARS}; c["SP"] = sp["cost_kEUR"]
        gap0 = c["DE0"] - c["SP"]
        rec = 100 * (c["DE0"] - c["DE_M1"]) / gap0 if gap0 > 1e-6 else 0.0
        rows.append(f"{TEX_NAME[pid]} & {c['DE0']:.1f} & {c['DE_M1']:.1f} & {c['DE_IND']:.1f} & {c['DE_M2']:.1f} & "
                    f"{c['DE_M3']:.1f} & {c['SP']:.1f} & {gap0:.2f} & {rec:.0f}\\% \\\\")
    _write_tex("tab_waterfall.tex",
               "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
               "District & DE0 & +Info & +Diag & +Pool & +Grid & SP & Gap & M1 rec. \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_prices(main):
    rows = []
    for pid in ORDER:
        r = main[pid]
        rows.append(f"{TEX_NAME[pid]} & {r['pi_mean']:.3f} & {r['tau_mean']:.4f} & "
                    f"{r.get('tau_cond_mean',0):.3f} & {r.get('tau_max',0):.3f} & {r['tau_bind_h']} \\\\")
    _write_tex("tab_prices.tex",
               "\\begin{tabular}{lrrrrr}\n\\toprule\n"
               "District & $\\bar\\pi$ & $\\bar\\tau$ (all h) & $\\bar\\tau\\,|\\,\\tau>0$ & $\\tau^{\\max}$ & $\\tau>0$ \\\\\n"
               " & [\\euro/kWh] & [\\euro/kWh] & [\\euro/kWh] & [\\euro/kWh] & [h/yr] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_equiv(main):
    rows = []
    for pid in ORDER:
        r = main[pid]; sp = r["SP"]; splp = r.get("SP_rel"); m3 = r["variants"]["DE_M3"]
        gap = 0.0 if abs(r.get("integrality_kEUR", 0.0)) < 5e-8 else r["integrality_kEUR"]
        simul = 0.0 if abs(r.get("simul_kWh", 0.0)) < 5e-8 else r["simul_kWh"]
        rows.append(f"{TEX_NAME[pid]} & {sp['cost_kEUR']:.3f} & {splp['cost_kEUR']:.3f} & "
                    f"{m3['cost_kEUR']:.3f} & {gap:.4f} & {simul:.4f} \\\\")
    _write_tex("tab_equiv.tex",
               "\\begin{tabular}{lrrrrr}\n\\toprule\n"
               "District & SP & SP$_{\\mathrm{rel}}$ & M3 & SP$-$SP$_{\\mathrm{rel}}$ & Simult.\\ ch/dis \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & [MWh] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def t_agent_ir():
    a = _load("agent_table.json")
    rows = [f"{r['agent']} & {r['de0_kEUR']:.2f} & {r['m3_kEUR']:.2f} & {r['delta_kEUR']:+.2f} & "
            f"{'yes' if r['ir_ok'] else 'no'} \\\\" for r in a["rows"]]
    comm = f"Community & {a['tot_de0']:.2f} & {a['tot_m3']:.2f} & {a['tot_m3']-a['tot_de0']:+.2f} & -- \\\\"
    _write_tex("tab_agent_ir.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrc@{}}\n\\toprule\n"
               "Agent & $\\mathrm{DE0}$ bill & $\\mathrm{M3}$ bill & Change & Individually \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & rational \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\midrule\n" + comm
               + "\n\\bottomrule\n\\end{tabular*}\n")


def t_marketpower():
    rows_in = _load("marketpower.json")["rows"]
    rows = [f"{TEX_NAME[r['pid']]} & {r['n_agents']} & {r['hhi']:.2f} & {r['export_share_binding']:.2f} & "
            f"{r['cong_payment_kEUR']:.2f} & {r['dev_gain_bound_pct']:.1f} \\\\" for r in rows_in]
    _write_tex("tab_marketpower.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}\n\\toprule\n"
               "District & Agents & HHI & Largest & Cong.\\ pay & Exposure \\\\\n"
               " & & & exp.\\ share & [k\\euro] & [\\% of bill] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_forecast():
    f = _load("forecast.json")
    rows = [f"{r['err_priv']:.2f} & {r['de0_kEUR']:.2f} & {r['m1_kEUR']:.2f} & {r['info_kEUR']:.2f} & "
            f"{r['info_pct_gap']:.1f} \\\\" for r in f["rows"]]
    _write_tex("tab_forecast.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrr@{}}\n\\toprule\n"
               "Private error s.d.\\ & DE0 cost & M1 cost & Info gain & M1 recovery \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [\\% of gap] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_uncertainty():
    u = _load("uncertainty.json")["stats"]
    pm = lambda k, d=2: f"{u[k]['mean']:.{d}f} $\\pm$ {u[k]['std']:.{d}f}"
    iqr = lambda k, d=2: f"[{u[k]['p25']:.{d}f}, {u[k]['p75']:.{d}f}]"
    rows = [
        ("Planner cost [k\\euro]", pm("cost_SP"), iqr("cost_SP")),
        ("Self-sufficiency SSR", pm("SSR", 3), iqr("SSR", 3)),
        ("Coordination gap [k\\euro]", pm("gap_kEUR"), iqr("gap_kEUR")),
        ("Coordination gap [\\% of DE0]", pm("gap_pct"), iqr("gap_pct")),
        ("Planner curtailment [MWh]", pm("curt_SP", 1), iqr("curt_SP", 1)),
        ("Deployable M1 recovery [\\% of gap]", pm("m1_recovery_pct", 1), iqr("m1_recovery_pct", 1)),
        ("Curtailment reduction at M3 [\\%]", pm("curt_reduction_pct", 1), iqr("curt_reduction_pct", 1)),
        ("Share: information (DE0$\\to$M1)", pm("share_info"), iqr("share_info")),
        ("Share: diagnostic residual (M1$\\to$IND)", pm("share_foresight"), iqr("share_foresight")),
        ("Share: P2P market (IND$\\to$M2)", pm("share_pool"), iqr("share_pool")),
        ("Share: grid price (M2$\\to$M3)", pm("share_grid"), iqr("share_grid")),
    ]
    _write_tex("tab_uncertainty.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrr@{}}\n\\toprule\n"
               f"Indicator ({int(u['n'])} draws) & Mean $\\pm$ s.d.\\ & IQR [p25, p75] \\\\\n\\midrule\n"
               + "\n".join(f"{k} & {m} & {q} \\\\" for k, m, q in rows)
               + f"\n\\midrule\nMechanism ranking preserved & {100*u['rank_preserved']:.0f}\\% of draws & -- \\\\"
               + "\n\\bottomrule\n\\end{tabular*}\n")


def t_fullyear():
    fy = _load("fullyear.json")
    a, b = fy["rep_week"], fy["full_year"]
    row = lambda lab, r: (f"{lab} & {r['cost_kEUR']:.1f} & {r['SSR']:.3f} & {r['LPI_os']:.3f} & "
                          f"{r['curt_MWh']:.0f} & {r['gap_kEUR']:.2f} & {r['tau_bind_h']:.0f} \\\\")
    err = lambda k: 100 * abs(b[k] - a[k]) / max(abs(a[k]), 1e-9)
    _write_tex("tab_fullyear.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\n"
               "Model & Cost & SSR & LPI$_{os}$ & Curtail & Gap & Cap-bind \\\\\n"
               " & [k\\euro] & & & [MWh] & [k\\euro] & [h/yr] \\\\\n\\midrule\n"
               + row("Four representative weeks", a) + "\n" + row("Full year (8736 h)", b) + "\n\\midrule\n"
               + f"Relative difference & {err('cost_kEUR'):.1f}\\% & {err('SSR'):.1f}\\% & {err('LPI_os'):.1f}\\% & "
               f"{err('curt_MWh'):.1f}\\% & {err('gap_kEUR'):.1f}\\% & {err('tau_bind_h'):.1f}\\% \\\\"
               + "\n\\bottomrule\n\\end{tabular*}\n")


def t_reg_sensitivity():
    audit = _load("convex_audit.json")
    rows = []
    for r in audit["regularization_sensitivity"]:
        m3 = r["M3"]
        rows.append(f"{r['epsilon']:.0e} & {m3['economic_cost_kEUR']:.3f} & {r['pool_increment_kEUR']:.3f} & "
                    f"{r['grid_increment_kEUR']:.3f} & {m3['pool_volume_MWh']:.1f} & "
                    f"{m3['pi_mean_EUR_per_kWh']:.4f} & {m3['tau_mean_EUR_per_kWh']:.4f} \\\\")
    _write_tex("tab_reg_sensitivity.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrrrr@{}}\n\\toprule\n"
               "$\\varepsilon^{\\mathrm r}$ & M3 cost & P2P increment & Grid increment & Pool volume & $\\bar\\pi$ & $\\bar\\tau$ \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [MWh] & [\\euro/kWh] & [\\euro/kWh] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_order_audit():
    a = _load("convex_audit.json")["order_audit"]
    pf, gf, sh = a["pool_first"], a["grid_first"], a["shapley"]
    rows = [f"Pool first & {pf['pool_kEUR']:.3f} & {pf['grid_kEUR']:.3f} & -- \\\\",
            f"Grid price first & {gf['pool_kEUR']:.3f} & {gf['grid_kEUR']:.3f} & -- \\\\",
            f"Two-order average (Shapley) & {sh['pool_kEUR']:.3f} & {sh['grid_kEUR']:.3f} & {a['interaction_kEUR']:.3f} \\\\"]
    _write_tex("tab_order_audit.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrr@{}}\n\\toprule\n"
               "Attribution rule & P2P market & Grid price & Interaction \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_provenance():
    rows = [
        ("Day-ahead price level and shape", "0.07--0.13 \\euro/kWh", "2023--24", "Public statistical", "ENTSO-E zonal day-ahead ranges"),
        ("Feed-in / export ratio", "0.45--0.55", "2023--24", "Assumed", "Retail feed-in vs.\\ import spread"),
        ("Use-of-system tariff", "0.028--0.040 \\euro/kWh", "2023--24", "Public statistical", "DSO volumetric network tariffs"),
        ("Grid emission factor", "0.03--0.42 kg/kWh", "2023", "Public statistical", "Zonal location-based factors"),
        ("PV specific yield", "950--1550 kWh/kWp", "TMY", "Public statistical", "PVGIS climate ranges"),
        ("Heat-pump COP curve", "3.2 at 7\\,$^\\circ$C, +0.10/$^\\circ$C, [1.8,5.5]", "--", "Assumed", "EN~14511 datasheet range"),
        ("Battery round-trip / SoC / C-rate", "0.90 / [0.15,0.85] / 0.25", "--", "Assumed", "Li-ion datasheet typical"),
        ("Battery degradation cost", "0.02 \\euro/kWh cycled", "--", "Assumed", "Levelized cycle-life estimate"),
        ("Thermal storage eff.\\ / loss / C-rate", "0.95 / 0.01\\,h$^{-1}$ / 0.5", "--", "Assumed", "Water-tank TES typical"),
        ("Shiftable load share / depth $\\alpha$ / window", "0.25 / 0.30 / 3 h", "--", "Assumed", "Demand-response literature range"),
        ("Forecast error s.d.\\ (private/shared)", "0.25 / 0.08", "--", "Assumed", "Day-ahead PV/load forecast skill"),
        ("Hosting limit (net export)", "130--260 kW", "--", "Assumed", "$\\approx$1/3 to 2/3 of a 400 kVA substation"),
        ("Installed capacities, floor areas", "per configuration", "--", "Calibrated", "Sized to positive-energy threshold"),
    ]
    _write_tex("tab_provenance.tex",
               "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{4.3cm}"
               ">{\\raggedright\\arraybackslash}p{3.5cm}cc X@{}}\n\\toprule\n"
               "Parameter family & Representative value & Base year & Status & Source type \\\\\n\\midrule\n"
               + "\n".join(f"{a} & {b} & {c} & {dd} & {e} \\\\" for a, b, c, dd, e in rows)
               + "\n\\bottomrule\n\\end{tabularx}\n")


def t_agents():
    rows = []
    for pid in DEEP:
        d = build_pilot_data(pid); p = C.PILOTS[pid]
        for a in p.agents:
            ag = d.agents[a.aid]
            tech = []
            if ag["pv_kwp"] > 0: tech.append(f"PV {ag['pv_kwp']:.0f}")
            if a.pvt_kwp > 0: tech.append(f"PVT {a.pvt_kwp:.0f}")
            if ag["hp"]: tech.append("HP")
            if ag["batt_kwh"] > 0: tech.append(f"batt {ag['batt_kwh']:.0f}")
            if ag["tes_kwh"] > 0: tech.append(f"TES {ag['tes_kwh']:.0f}")
            if ag["dh"]: tech.append("DH")
            rows.append(f"{TEX_NAME[pid]} & {a.aid} & {a.archetype.replace('_',' ')} & "
                        f"{C.ARCH[a.archetype].floor_m2:.0f} & {', '.join(tech)} \\\\")
    _write_tex("tab_agents.tex",
               "\\begin{tabularx}{\\textwidth}{@{}lll r X@{}}\n\\toprule\n"
               "District & Agent & Archetype & Floor [m$^2$] & Technologies (kWp, kWh) \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_zones():
    znice = {"DK2": "Denmark (DK2)", "SE3": "Sweden (SE3)", "IE": "Ireland (SEM)",
             "DE": "Germany", "AT": "Austria", "TR": "Turkey", "RO": "Romania"}
    rows = [f"{znice.get(zk, zk)} & {z.price_mean:.3f} & {z.price_amp:.3f} & {z.export_ratio:.2f} & "
            f"{z.tariff_use:.3f} & {z.ef_mean:.3f} & {z.ef_amp:.3f} & {z.ef_dh:.3f} \\\\"
            for zk, z in C.ZONES.items()]
    _write_tex("tab_zones.tex",
               "\\begin{tabularx}{\\textwidth}{@{}l*{7}{>{\\centering\\arraybackslash}X}@{}}\n\\toprule\n"
               "Zone & $\\bar\\lambda^{\\mathrm{imp}}$ & amp. & $\\lambda^{\\mathrm{exp}}/\\lambda^{\\mathrm{imp}}$ & "
               "$\\tau^{\\mathrm{use}}$ & $\\overline{\\mathrm{EF}}^{\\mathrm{g}}$ & amp. & $\\mathrm{EF}^{\\mathrm{dh}}$ \\\\\n"
               " & [\\euro/kWh] & [\\euro/kWh] & & [\\euro/kWh] & [kg/kWh] & [kg/kWh] & [kg/kWh] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_tech():
    y = C.PV_YIELD
    rows = [
        ("Heat-pump COP intercept / slope", f"{C.COP0:.1f} at 7\\,$^\\circ$C / {C.KAPPA_COP:.2f} per $^\\circ$C"),
        ("Battery round-trip efficiency", f"{C.ETA_CH*C.ETA_DIS:.2f}"),
        ("Battery state-of-charge window", f"[{C.SOC_MIN:.2f}, {C.SOC_MAX:.2f}]"),
        ("Battery power-to-energy ratio", f"{C.BATT_C_RATE:.2f} (C/{int(1/C.BATT_C_RATE)})"),
        ("Battery degradation cost", f"{C.DEG_COST:.3f} \\euro/kWh cycled"),
        ("Heat-pump COP bounds", f"[{C.COP_MIN:.1f}, {C.COP_MAX:.1f}]"),
        ("Thermal storage efficiency / standing loss", f"{C.ETA_TES*C.ETA_TES:.2f} / {C.KAPPA_TES:.2f} per h"),
        ("Thermal storage power-to-energy ratio", f"{C.TES_C_RATE:.2f}"),
        ("Shiftable-load flexibility fraction $\\alpha_a$", f"{C.ALPHA_SHIFT:.2f}, window {C.W_SHIFT} h"),
        ("District-heating price", "0.085 \\euro/kWh"),
        ("Carbon price (base)", "100 \\euro/t"),
        ("PV specific yield (DK/IE/SE/DE/AT/TR/RO)",
         f"{y['DK']}/{y['IE']}/{y['SE']}/{y['DE']}/{y['AT']}/{y['TR']}/{y['RO']} kWh/kWp/yr"),
    ]
    _write_tex("tab_tech.tex",
               "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{6.2cm}X@{}}\n\\toprule\n"
               "Parameter & Value \\\\\n\\midrule\n"
               + "\n".join(f"{k} & {v} \\\\" for k, v in rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def t_recycling():
    d = _load("recycling.json")
    rows = [f"{r['agent']} & {r['de0']:.2f} & {r['m3']:.2f} & {r['m3_eq']:.2f} & {r['m3_prop']:.2f} & "
            f"{'yes' if r['ir_eq'] else 'no'} & {'yes' if r['ir_prop'] else 'no'} \\\\" for r in d["rows"]]
    _write_tex("tab_recycling.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrcc@{}}\n\\toprule\n"
               "Agent & $\\mathrm{DE0}$ & $\\mathrm{M3}$ & $\\mathrm{M3}$+equal & $\\mathrm{M3}$+prop.\\ & IR equal & IR prop.\\ \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [k\\euro] & & \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_reg_deployable():
    rows_in = _load("regularization_deployable.json")
    rows = [f"$10^{{{int(round(np.log10(r['epsilon'])))}}}$ & {r['de0_kEUR']:.3f} & {r['m1_kEUR']:.3f} & "
            f"{r['info_kEUR']:.3f} & {r['m1_recovery_pct']:.1f} \\\\" for r in rows_in]
    _write_tex("tab_reg_deployable.tex",
               "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrr@{}}\n\\toprule\n"
               "$\\varepsilon^{\\mathrm r}$ & DE0 cost & M1 cost & Info gain & M1 recovery \\\\\n"
               " & [k\\euro] & [k\\euro] & [k\\euro] & [\\% of gap] \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n")


def t_montecarlo():
    rows = [
        ("Ambient temperature", "additive $\\mathcal N(0,1.2\\,^\\circ\\mathrm{C})$ per hour", "common to all agents"),
        ("Solar clearness", "$\\mathrm{Beta}(5,2)$ per hour", "common, clipped to $[0.2,1.0]$"),
        ("Electrical load", "multiplicative $1+\\mathcal N(0,0.05)$ per hour", "independent per agent"),
        ("Day-ahead price", "additive $\\mathcal N(0,0.01)$~\\euro/kWh", "common, floored at 0.01"),
        ("Emission factor", "additive $\\mathcal N(0,0.01)$~kg/kWh", "common, floored at 0.005"),
        ("Private forecast error", "multiplicative $1+\\mathcal N(0,0.25)$", "independent per agent, clipped $\\ge0.2$"),
        ("Shared forecast error", "multiplicative $1+\\mathcal N(0,0.08)$", "common to all agents, clipped $\\ge0.2$"),
    ]
    _write_tex("tab_montecarlo.tex",
               "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{4.0cm}"
               ">{\\raggedright\\arraybackslash}p{3.7cm}X@{}}\n\\toprule\n"
               "Quantity & Perturbation & Correlation and bounds \\\\\n\\midrule\n"
               + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows) + "\n\\bottomrule\n\\end{tabularx}\n")


def generate_tables():
    main = _load("main_results.json")
    t_pilots(main); t_main(main); t_waterfall(main); t_prices(main); t_equiv(main)
    t_agents(); t_zones(); t_tech(); t_provenance(); t_montecarlo()
    for name, fn in [("agent_ir", t_agent_ir), ("marketpower", t_marketpower), ("forecast", t_forecast),
                     ("uncertainty", t_uncertainty), ("fullyear", t_fullyear),
                     ("reg_sensitivity", t_reg_sensitivity), ("order_audit", t_order_audit),
                     ("recycling", t_recycling), ("reg_deployable", t_reg_deployable)]:
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print(f"  (tab_{name} skipped: {e})")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_geomap():
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    src = os.path.join(GEO, "countries_50m.geojson")
    if not os.path.exists(src):
        src = os.path.join(GEO, "countries_110m.geojson")
    geo = json.load(open(src, encoding="utf-8"))
    lon0, lon1, lat0, lat1 = -12, 42, 34, 69
    patches = []
    for f in geo["features"]:
        g = f.get("geometry")
        if not g:
            continue
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ext = poly[0]
            xs = [p[0] for p in ext]; ys = [p[1] for p in ext]
            if max(xs) < lon0 - 6 or min(xs) > lon1 + 6 or max(ys) < lat0 - 6 or min(ys) > lat1 + 6:
                continue
            patches.append(MplPoly(list(zip(xs, ys)), closed=True))
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.set_facecolor("#dbe9f3")
    ax.set_xticks(range(-10, 41, 10)); ax.set_yticks(range(35, 66, 10))
    ax.grid(True, color="#c6d6e2", lw=0.4, ls=(0, (1, 3)), zorder=0)
    ax.add_collection(PatchCollection(patches, facecolor="#eef0e9", edgecolor="#9fabb3",
                                      linewidths=0.35, zorder=1, joinstyle="round"))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(52)))
    ax.tick_params(labelsize=6, length=2.5, color="#9fabb3", labelcolor="#5b6a73")
    ax.set_xticklabels([f"{abs(x)}$^\\circ${'W' if x < 0 else 'E'}" if x else "0$^\\circ$" for x in range(-10, 41, 10)])
    ax.set_yticklabels([f"{y}$^\\circ$N" for y in range(35, 66, 10)])
    for s in ax.spines.values():
        s.set_edgecolor("#9fabb3"); s.set_linewidth(0.6)
    NAVY, CORAL = "#14507d", "#e67e22"
    sites = [
        (-8.62, 52.66, "Limerick (IE)", "P2P market:\nlocal trading,\n+self-consumption", "deep", (-11.5, 44.0)),
        (12.48, 55.79, "Virum (DK)", "Sector coupling:\nPV + heat pumps\n+ district heating", "deep", (2.0, 62.5)),
        (16.55, 59.61, "Västerås (SE)", "PV facades +\nHVAC flexibility", "deep", (20.0, 65.5)),
        (13.40, 52.52, "Berlin (DE)", "", "deep", None),
        (15.44, 47.07, "Graz (AT)", "Power-to-heat on\ndistrict heating (DE-AT)", "deep", (2.5, 40.0)),
        (26.10, 44.43, "Bucharest (RO)", "Smart HVAC on\ndistrict heating", "illus", (33.5, 51.0)),
        (32.85, 39.93, "Ankara (TR)", "Solar-assisted\nheat pumps + PVT", "illus", (36.5, 43.5)),
    ]
    ax.plot([13.40, 15.44], [52.52, 47.07], color=NAVY, lw=1.0, ls=":", zorder=3)
    for lon, lat, label, role, tier, txy in sites:
        deep = tier == "deep"
        ax.scatter([lon], [lat], s=95 if deep else 80, marker="*" if deep else "o",
                   facecolor=NAVY if deep else "white", edgecolor=NAVY if deep else CORAL,
                   linewidths=1.6, zorder=5)
        ax.annotate(label, (lon, lat), xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, fontweight="bold", color="#22303a", zorder=6)
        if txy is not None:
            col = NAVY if deep else CORAL
            ax.annotate(role, xy=(lon, lat), xytext=txy, fontsize=6.8, color="#22303a",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc=("#eaf1f7" if deep else "#fdf0e3"), ec=col, lw=0.9),
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8, connectionstyle="arc3,rad=0.1"), zorder=6)
    ax.scatter([], [], marker="*", s=95, facecolor=NAVY, edgecolor=NAVY, label="Deep configuration")
    ax.scatter([], [], marker="o", s=80, facecolor="white", edgecolor=CORAL, label="Illustrative configuration")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    _save_fig(fig, "F0_testbed_map")


def fig_concept(main):
    d = main["virum"]["series"]
    I = np.array(d["I"]); E = np.array(d["E"]); CT = np.array(d["CURT"]); PV = np.array(d["PV"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), sharey=True)
    for ax, (a, b, ttl) in zip(axes, [(0, 168, "Winter week (deficit-dominated)"),
                                       (336, 504, "Summer week (surplus-dominated)")]):
        h = np.arange(b - a)
        ax.fill_between(h, 0, PV[a:b], color="#f4c430", alpha=.55, label="PV generation")
        ax.plot(h, I[a:b], color="#c1121f", lw=1.0, label="grid import (deficit)")
        ax.plot(h, -(E[a:b] + CT[a:b]), color="#0353a4", lw=1.0, label="export + curtailment (surplus)")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("hour [h]"); ax.set_title(ttl, fontsize=8)
    axes[0].set_ylabel("power [kW]")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)
    _save_fig(fig, "F1_concept")


def fig_waterfall(main):
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.8))
    for ax, pid in zip(axes, DEEP):
        v = main[pid]["variants"]; sp = main[pid]["SP"]
        costs = [v[k]["cost_kEUR"] for k in VARS] + [sp["cost_kEUR"]]
        x = np.arange(len(costs))
        ax.bar(x, costs, color=["#8d99ae", "#6c8ead", "#4a7ba6", "#2a6f97", "#014f86", "#013a63"])
        ax.set_xticks(x); ax.set_xticklabels(VLAB + ["SP\nplanner"], fontsize=6)
        ax.set_title(FIG_NAME[pid], fontsize=8)
        ax.set_ylim(min(costs) * 0.9, max(costs) * 1.02)
        for xi, c in zip(x, costs):
            ax.text(xi, c, f"{c:.1f}", ha="center", va="bottom", fontsize=6)
    axes[0].set_ylabel("annual net cost [k€]")
    _save_fig(fig, "F2_waterfall")


def fig_prices(main):
    d = main["virum"]["series"]
    pi = np.array(d["pi"]); tau = np.array(d["tau"])
    nw = len(tau) // 168
    binding = [int(np.sum(tau[wk * 168:(wk + 1) * 168] > 1e-4)) for wk in range(nw)]
    wk = int(np.argmax(binding)); a = wk * 168; b = a + 168
    h = np.arange(168)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 2.9))
    ax.plot(h, pi[a:b], color="#2a6f97", lw=1.1, label=r"P2P clearing price $\pi$")
    ax.plot(h, tau[a:b], color="#c1121f", lw=1.0, label=r"congestion price $\tau$")
    ax.fill_between(h, 0, tau[a:b], color="#c1121f", alpha=.18)
    ax.set_xlabel("hour [h]"); ax.set_ylabel("price [€/kWh]")
    ax.set_title("Prices in the binding week", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False)
    wa = _annual_weights(len(tau))
    order = np.argsort(tau)[::-1]
    xh = np.cumsum(wa[order])
    ax2.plot(xh, tau[order], color="#c1121f", lw=1.2)
    ax2.fill_between(xh, 0, tau[order], color="#c1121f", alpha=.18)
    ax2.set_xlabel("annual hours [h]"); ax2.set_ylabel(r"$\tau$ [€/kWh]")
    ax2.set_title("Annual congestion-price duration", fontsize=8)
    _save_fig(fig, "F4_prices")


def fig_pareto(par):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for pid in par:
        pts = sorted([p for p in par[pid] if 0.0 <= p["SSR"] <= 1.0], key=lambda p: p["SSR"])
        if not pts:
            continue
        ss = [p["SSR"] for p in pts]; cost = [p["cost_kEUR"] for p in pts]
        if len(pts) == 1:
            line, = ax.plot(ss, cost, marker="o", ms=6, lw=0, label=FIG_NAME.get(pid, pid) + " (wall at optimum)")
            ax.annotate("wall at optimum", (ss[0], cost[0]), xytext=(6, -2),
                        textcoords="offset points", fontsize=6.5, color=line.get_color(), va="center")
        else:
            line, = ax.plot(ss, cost, marker="o", ms=4, lw=1.4, label=FIG_NAME.get(pid, pid))
        ax.plot(ss[-1], cost[-1], marker="*", ms=13, mec="k", mew=0.5, color=line.get_color(), zorder=5)
    ax.set_xlabel("self-sufficiency [-]"); ax.set_ylabel("annual net cost [k€]")
    ax.legend(fontsize=7, frameon=False)
    _save_fig(fig, "F5_selfsuff_cost")


def fig_sensitivity(sens):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.0))
    H = sens["hosting"]; s = [r["scale"] for r in H]
    tot = [r["DE0"]["cost_kEUR"] - r["SP"]["cost_kEUR"] for r in H]
    res = [r["DE_M3"]["cost_kEUR"] - r["SP"]["cost_kEUR"] for r in H]
    a1.plot(s, tot, marker="o", ms=3, color="#014f86", label="total gap (DE0 - SP)")
    a1.plot(s, res, marker="s", ms=3, color="#2a9d8f", label="residual after platform (M3 - SP)")
    a1.fill_between(s, res, tot, color="#8ecae6", alpha=.35, label="recovered by platform")
    a1.set_xlabel("hosting limit / base [-]"); a1.set_ylabel("coordination gap [k€]")
    a1.set_title("Gap vs grid scarcity", fontsize=8); a1.legend(fontsize=6, frameon=False)
    Cb = sens["carbon"]; cp = [r["carbon"] * 1000 for r in Cb]
    a2.plot(cp, [r["DE0"]["curt_MWh"] for r in Cb], marker="o", ms=3, color="#c1121f", label="DE0")
    a2.plot(cp, [r["DE_M3"]["curt_MWh"] for r in Cb], marker="s", ms=3, color="#2a9d8f", label="M3 grid-aware")
    a2.set_xlabel("carbon price [€/t]"); a2.set_ylabel("curtailment [MWh]")
    a2.set_title("Curtailment vs carbon price", fontsize=8); a2.legend(fontsize=6.5, frameon=False)
    St = sens["storage"]; ss = [r["scale"] for r in St]
    a3.plot(ss, [r["DE0"]["SSR"] for r in St], marker="o", ms=3, label="DE0")
    a3.plot(ss, [r["DE_M3"]["SSR"] for r in St], marker="s", ms=3, label="M3")
    a3.plot(ss, [r["SP"]["SSR"] for r in St], marker="^", ms=3, label="SP")
    a3.set_xlabel("storage capacity / base [-]"); a3.set_ylabel("self-sufficiency [-]")
    a3.set_title("Self-sufficiency vs storage", fontsize=8); a3.legend(fontsize=6.5, frameon=False)
    _save_fig(fig, "F6_sensitivity")


def fig_sensitivity2(sens):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.0))
    R = sens["res"]; r = [x["ratio"] for x in R]
    a1.plot(r, [x["SP"]["SSR"] for x in R], marker="o", ms=3, color="#014f86", label="self-sufficiency")
    a1.plot(r, [x["SP"]["LPI_os"] for x in R], marker="s", ms=3, color="#c1121f", label="oversupply")
    a1.set_xlabel("renewable penetration [-]"); a1.set_ylabel("fraction [-]")
    a1.set_title("Self-sufficiency and oversupply vs RES", fontsize=8); a1.legend(fontsize=6.5, frameon=False)
    F = sens["flex"]; al = [x["alpha"] for x in F]
    a2.plot(al, [x["DE0"]["SSR"] for x in F], marker="o", ms=3, label="DE0")
    a2.plot(al, [x["DE_M3"]["SSR"] for x in F], marker="s", ms=3, label="M3")
    a2.plot(al, [x["SP"]["SSR"] for x in F], marker="^", ms=3, label="SP")
    a2.set_xlabel("demand-response depth [-]"); a2.set_ylabel("self-sufficiency [-]")
    a2.set_title("Self-sufficiency vs DR depth", fontsize=8); a2.legend(fontsize=6.5, frameon=False)
    Fi = sens["feedin"]; fr = [x["ratio"] for x in Fi]
    a3.plot(fr, [x["DE0"]["cost_kEUR"] - x["SP"]["cost_kEUR"] for x in Fi], marker="o", ms=3, color="#014f86")
    a3.set_xlabel("feed-in / import price [-]"); a3.set_ylabel("coordination gap [k€]")
    a3.set_title("Gap vs feed-in price", fontsize=8)
    _save_fig(fig, "F7_sensitivity2")


def fig_energy_balance(main):
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    x = np.arange(len(ORDER)); wbar = 0.36; ymax = 0.0
    for i, pid in enumerate(ORDER):
        sp = main[pid]["SP"]
        pv, imp, exp, curt = sp["pv_MWh"], sp["imp_MWh"], sp["exp_MWh"], sp["curt_MWh"]
        sc = max(pv - exp - curt, 0.0)
        ax.bar(i - 0.20, sc, wbar, color="#2a9d8f", zorder=3, label="self-consumed RES" if i == 0 else "")
        ax.bar(i - 0.20, exp, wbar, bottom=sc, color="#2a6f97", zorder=3, label="exported" if i == 0 else "")
        ax.bar(i - 0.20, curt, wbar, bottom=sc + exp, color="#c1121f", zorder=3, label="curtailed" if i == 0 else "")
        ax.bar(i + 0.20, sc, wbar, color="#2a9d8f", zorder=3)
        ax.bar(i + 0.20, imp, wbar, bottom=sc, color="#e9a13b", zorder=3, label="imported" if i == 0 else "")
        ymax = max(ymax, sc + exp + curt, sc + imp)
    ax.set_xticks(x); ax.set_xticklabels([FIG_NAME[p] for p in ORDER], fontsize=8)
    ax.set_ylabel("annual energy [MWh]"); ax.set_ylim(0, ymax * 1.30); ax.margins(x=0.02)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    _save_fig(fig, "F8_energy_balance")


def fig_last20(main):
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    x = np.arange(len(ORDER))
    for i, p in enumerate(ORDER):
        sp = main[p]["SP"]
        deficit = sp["imp_MWh"] / max(sp["dem_MWh"], 1e-9)
        exp_f = sp["exp_MWh"] / max(sp["pv_MWh"], 1e-9)
        curt_f = sp["curt_MWh"] / max(sp["pv_MWh"], 1e-9)
        ax.bar(i, exp_f, 0.6, color="#2a6f97", zorder=3, label="exported" if i == 0 else "")
        ax.bar(i, curt_f, 0.6, bottom=exp_f, color="#c1121f", zorder=3, label="curtailed" if i == 0 else "")
        ax.bar(i, -deficit, 0.6, color="#e9a13b", zorder=3, label="imported (deficit)" if i == 0 else "")
        ax.text(i, exp_f + curt_f + 0.01, f"{exp_f + curt_f:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i, -deficit - 0.01, f"{deficit:.2f}", ha="center", va="top", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([FIG_NAME[p] for p in ORDER], fontsize=7.5)
    ax.set_ylabel("fraction [-]"); ax.set_ylim(-0.52, 0.60)
    ax.legend(fontsize=7, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    _save_fig(fig, "F9_last20")


def fig_seasonal(main):
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), sharey=True)
    for ax, pid in zip(axes, DEEP):
        s = main[pid]["series"]
        I = np.array(s["I"]); E = np.array(s["E"]); CT = np.array(s["CURT"])
        dem = np.array(s["dem"]); PV = np.array(s["PV"])
        n = len(I) // 4
        ss, os_ = [], []
        for k in range(4):
            sl = slice(k * n, (k + 1) * n)
            ss.append(1 - I[sl].sum() / max(dem[sl].sum(), 1e-9))
            os_.append((E[sl].sum() + CT[sl].sum()) / max(PV[sl].sum(), 1e-9))
        xk = np.arange(4)
        ax.bar(xk - 0.2, ss, 0.4, color="#2a9d8f", label="self-sufficiency")
        ax.bar(xk + 0.2, os_, 0.4, color="#2a6f97", label="oversupply")
        ax.set_xticks(xk); ax.set_xticklabels(["W", "Sp", "Su", "Au"], fontsize=7)
        ax.set_title(FIG_NAME[pid], fontsize=8); ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction [-]"); axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    _save_fig(fig, "F10_seasonal")


def fig_uncertainty(u):
    draws = u["draws"]; st = u["stats"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.0))
    gaps = [d["gap_pct"] for d in draws]
    ax.hist(gaps, bins=12, color="#2a6f97", alpha=.8)
    ax.axvline(st["gap_pct"]["mean"], color="#c1121f", lw=1.2, label=f"mean {st['gap_pct']['mean']:.1f}%")
    ax.set_xlabel("coordination gap [% of DE0 cost]"); ax.set_ylabel("count")
    ax.set_title("Gap distribution across seeds", fontsize=8); ax.legend(fontsize=7, frameon=False)
    labels = ["information", "diagnostic", "P2P market", "grid price"]
    keys = ["share_info", "share_foresight", "share_pool", "share_grid"]
    means = [st[k]["mean"] for k in keys]; sds = [st[k]["std"] for k in keys]
    ax2.bar(np.arange(4), means, 0.6, yerr=sds, capsize=3, color=["#8d99ae", "#6c8ead", "#2a6f97", "#014f86"])
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("share [-]"); ax2.set_title("Mechanism shares (mean $\\pm$ s.d.)", fontsize=8)
    _save_fig(fig, "F12_uncertainty")


def generate_figures():
    try:
        fig_geomap()
    except FileNotFoundError:
        print("  (F0 skipped: data/countries_*.geojson missing)")
    main = _load("main_results.json")
    fig_concept(main); fig_waterfall(main); fig_prices(main)
    fig_energy_balance(main); fig_last20(main); fig_seasonal(main)
    for name, loader, fn in [("pareto.json", "pareto", fig_pareto),
                             ("sensitivity_virum.json", "sensitivity", fig_sensitivity),
                             ("sensitivity_virum.json", "sensitivity2", fig_sensitivity2),
                             ("uncertainty.json", "uncertainty", fig_uncertainty)]:
        try:
            fn(_load(name))
        except FileNotFoundError:
            print(f"  ({loader} skipped: {name} missing)")


if __name__ == "__main__":
    print("Tables:")
    generate_tables()
    print("Figures:")
    generate_figures()
    print(f"Tables in {TABLES}\nFigures in {FIGURES}")

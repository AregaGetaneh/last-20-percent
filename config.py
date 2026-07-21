"""Model configuration: run settings, technology parameters, zonal market
calibration, building archetypes, and the district-community definitions.

Units: power kW, energy kWh, temperature deg C, price EUR/kWh, emission factor
kg CO2/kWh. One representative step equals DT hours.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# Run settings
SEED = 42
THREADS = 1
MIP_GAP = 1e-4

DT = 1.0
SEASONS = ["winter", "spring", "summer", "autumn"]
HOURS_PER_SEASON = 168                                   # one representative week per season
SEASON_WEIGHT = {"winter": 90, "spring": 92, "summer": 92, "autumn": 91}   # days per season

# Heat pump
COP0 = 3.2                                               # COP at 7 deg C (EN 14511)
KAPPA_COP = 0.10                                         # COP slope per deg C
COP_MIN, COP_MAX = 1.8, 5.5

# Battery
ETA_CH = 0.90 ** 0.5                                     # one-way charge efficiency (round trip 0.90)
ETA_DIS = 0.90 ** 0.5
SOC_MIN, SOC_MAX = 0.15, 0.85
BATT_C_RATE = 0.25                                       # power / energy capacity
DEG_COST = 0.02                                          # degradation cost on cycled energy [EUR/kWh]

# Thermal storage
ETA_TES = 0.95 ** 0.5
KAPPA_TES = 0.01                                         # standing loss per hour [1/h]
TES_C_RATE = 0.5

# Demand response
ALPHA_SHIFT = 0.30                                       # per-hour shiftable-load deviation fraction
W_SHIFT = 3                                              # shift window [h]

PV_YIELD = {                                             # specific yield [kWh/kWp/yr]
    "DK": 980, "IE": 950, "SE": 950, "DE": 1050, "AT": 1080, "TR": 1550, "RO": 1300,
}


@dataclass
class ZoneMarket:
    zone: str
    price_mean: float                                   # mean day-ahead price [EUR/kWh]
    price_amp: float                                    # intraday amplitude [EUR/kWh]
    export_ratio: float                                 # export price as a fraction of import price
    tariff_use: float                                   # volumetric use-of-system tariff [EUR/kWh]
    ef_mean: float                                      # mean grid emission factor [kg/kWh]
    ef_amp: float                                       # intraday emission-factor amplitude [kg/kWh]
    ef_dh: float                                        # district-heating emission factor [kg/kWh]


ZONES: Dict[str, ZoneMarket] = {
    "DK2": ZoneMarket("DK2", 0.095, 0.045, 0.55, 0.030, 0.110, 0.060, 0.045),
    "SE3": ZoneMarket("SE3", 0.070, 0.035, 0.55, 0.028, 0.030, 0.015, 0.030),
    "IE":  ZoneMarket("IE",  0.130, 0.050, 0.50, 0.035, 0.300, 0.080, 0.150),
    "DE":  ZoneMarket("DE",  0.120, 0.060, 0.50, 0.040, 0.350, 0.120, 0.180),
    "AT":  ZoneMarket("AT",  0.115, 0.055, 0.50, 0.038, 0.160, 0.070, 0.120),
    "TR":  ZoneMarket("TR",  0.090, 0.040, 0.45, 0.030, 0.420, 0.090, 0.200),
    "RO":  ZoneMarket("RO",  0.100, 0.045, 0.50, 0.032, 0.290, 0.080, 0.160),
}


@dataclass
class Archetype:
    kind: str                                           # residential | commercial | office | public
    base_load_kw: float                                 # mean electrical base load [kW]
    heat_kw_per_m2: float                               # design heat load density [kW/m2]
    floor_m2: float                                     # heated floor area [m2]
    occ_morning: float                                  # occupancy weight shaping load (0..1)
    occ_evening: float


ARCH = {
    "res_block":  Archetype("residential", 22.0, 0.045, 3200, 0.5, 1.0),
    "res_small":  Archetype("residential",  6.0, 0.050,  900, 0.4, 1.0),
    "commercial": Archetype("commercial",  35.0, 0.035, 2500, 0.9, 0.4),
    "office":     Archetype("office",      28.0, 0.030, 2200, 1.0, 0.3),
    "public":     Archetype("public",      18.0, 0.040, 1800, 0.8, 0.5),
}


@dataclass
class Agent:
    aid: str
    archetype: str
    pv_kwp: float = 0.0
    pvt_kwp: float = 0.0                                 # photovoltaic-thermal peak [kW]
    batt_kwh: float = 0.0
    hp: bool = False
    tes_kwh: float = 0.0
    dh: bool = False                                    # district-heating intake
    flex: bool = True                                   # shiftable-load capable

    @property
    def batt_kw(self) -> float:
        return self.batt_kwh * BATT_C_RATE

    @property
    def tes_kw(self) -> float:
        return self.tes_kwh * TES_C_RATE


@dataclass
class Pilot:
    pid: str
    name: str
    country: str
    zone: str
    tier: str                                           # deep | illustrative
    agents: List[Agent]
    hosting_limit_kw: float                             # DSO net-export hosting limit at the coupling point
    ped_ratio: float = 1.05                             # target annual renewable / annual electricity-equivalent demand
    has_dh_network: bool = False

    @property
    def pv_total(self) -> float:
        return sum(a.pv_kwp for a in self.agents)

    @property
    def batt_total(self) -> float:
        return sum(a.batt_kwh for a in self.agents)


# Deep configurations
VIRUM = Pilot(
    pid="virum", name="Sector-coupled social-housing district", country="DK", zone="DK2",
    tier="deep", hosting_limit_kw=180.0, has_dh_network=True,
    agents=[
        Agent("VI-A", "res_block", pv_kwp=55, batt_kwh=130, hp=True, tes_kwh=220, dh=True),
        Agent("VI-B", "res_block", pv_kwp=45, batt_kwh=110, hp=True, tes_kwh=180, dh=True),
        Agent("VI-C", "res_block", pv_kwp=30, batt_kwh=60,  hp=True, tes_kwh=120, dh=True),
        Agent("VI-D", "public",    pv_kwp=20, batt_kwh=29,  hp=True, tes_kwh=80,  dh=True),
    ],
)

LIMERICK = Pilot(
    pid="limerick", name="Mixed-use peer-to-peer district", country="IE", zone="IE",
    tier="deep", hosting_limit_kw=260.0, ped_ratio=1.10, has_dh_network=False,
    agents=[
        Agent("LK-C1", "commercial", pv_kwp=60, batt_kwh=80, hp=True, tes_kwh=40),
        Agent("LK-C2", "commercial", pv_kwp=40, batt_kwh=40, hp=True, tes_kwh=30),
        Agent("LK-R1", "res_block",  pv_kwp=25, batt_kwh=30, hp=True, tes_kwh=60),
        Agent("LK-R2", "res_small",  pv_kwp=8,  batt_kwh=10, hp=True, tes_kwh=25),
        Agent("LK-R3", "res_small",  pv_kwp=6,  batt_kwh=0,  hp=True, tes_kwh=20),
        Agent("LK-P",  "public",     pv_kwp=15, batt_kwh=20, hp=True, tes_kwh=30),
    ],
)

VASTERAS = Pilot(
    pid="vasteras", name="PV-facade office district", country="SE", zone="SE3",
    tier="deep", hosting_limit_kw=160.0, has_dh_network=True,
    agents=[
        Agent("VA-O1", "office", pv_kwp=50, batt_kwh=60, hp=True, tes_kwh=100, dh=True),
        Agent("VA-O2", "office", pv_kwp=35, batt_kwh=40, hp=True, tes_kwh=70,  dh=True),
        Agent("VA-C",  "commercial", pv_kwp=30, batt_kwh=30, hp=False, dh=True),
        Agent("VA-R",  "res_block",  pv_kwp=20, batt_kwh=25, hp=True, tes_kwh=50, dh=True),
    ],
)

GRAZ = Pilot(
    pid="graz", name="Power-to-heat district-heating community", country="DE", zone="DE",
    tier="deep", hosting_limit_kw=200.0, has_dh_network=True,
    agents=[
        Agent("GB-1", "res_block", pv_kwp=30, batt_kwh=20, hp=True, tes_kwh=200, dh=True),
        Agent("GB-2", "res_block", pv_kwp=25, batt_kwh=20, hp=True, tes_kwh=180, dh=True),
        Agent("GB-3", "public",    pv_kwp=20, batt_kwh=15, hp=True, tes_kwh=150, dh=True),
        Agent("GB-4", "commercial", pv_kwp=25, batt_kwh=20, hp=True, tes_kwh=160, dh=True),
    ],
)

# Illustrative configurations
TURKEY = Pilot(
    pid="turkey", name="Solar-assisted heat-pump district", country="TR", zone="TR",
    tier="illustrative", hosting_limit_kw=160.0, ped_ratio=1.15, has_dh_network=False,
    agents=[
        Agent("TR-1", "res_block", pv_kwp=40, pvt_kwp=20, batt_kwh=30, hp=True, tes_kwh=120),
        Agent("TR-2", "res_small", pv_kwp=10, pvt_kwp=6,  batt_kwh=8,  hp=True, tes_kwh=40),
        Agent("TR-3", "public",    pv_kwp=25, batt_kwh=15, hp=True, tes_kwh=60),
    ],
)

ROMANIA = Pilot(
    pid="romania", name="Smart-HVAC office district", country="RO", zone="RO",
    tier="illustrative", hosting_limit_kw=130.0, ped_ratio=1.00, has_dh_network=True,
    agents=[
        Agent("RO-1", "office", pv_kwp=25, batt_kwh=20, hp=True, tes_kwh=50, dh=True),
        Agent("RO-2", "commercial", pv_kwp=20, batt_kwh=15, hp=False, dh=True),
        Agent("RO-3", "public", pv_kwp=15, batt_kwh=10, hp=True, tes_kwh=40, dh=True),
    ],
)

PILOTS: Dict[str, Pilot] = {p.pid: p for p in [VIRUM, LIMERICK, VASTERAS, GRAZ, TURKEY, ROMANIA]}
DEEP_PILOTS = [p.pid for p in PILOTS.values() if p.tier == "deep"]


if __name__ == "__main__":
    for p in PILOTS.values():
        print(f"{p.pid:9s} [{p.tier:12s}] {len(p.agents)} agents, "
              f"PV {p.pv_total:.0f} kWp, battery {p.batt_total:.0f} kWh, zone {p.zone}")

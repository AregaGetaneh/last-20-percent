# Calibration provenance

All inputs are calibration archetypes generated deterministically by `data.py` and
`config.py`. They are not metered field data. The values below are stylized
2023-24 calibration figures inspired by the named sources; they are not a
statement of current 2026 remuneration or tariffs. Each figure is reproducible
from `config.py` (`SEED = 42`).

| Parameter family | Value / range | Basis | Source organization | Transformation in the code |
|---|---|---|---|---|
| Day-ahead price level and shape | 0.07-0.13 EUR/kWh, diurnal amplitude | 2023-24 | ENTSO-E Transparency Platform (zonal day-ahead) | zonal mean plus sinusoidal diurnal shape, `config.ZONES`, `data.build_pilot_data` |
| Export / feed-in ratio | 0.45-0.55 of import price | 2023-24 | National schemes then in force (see below) | `ZoneMarket.export_ratio`, applied as `price_exp = ratio * price_imp` |
| Use-of-system tariff | 0.028-0.040 EUR/kWh | 2023-24 | Published DSO volumetric network tariffs | `ZoneMarket.tariff_use`, flat volumetric charge |
| Grid emission factor | 0.03-0.42 kg/kWh, diurnal | 2023 | Zonal location-based intensities (attributional) | `ZoneMarket.ef_mean/ef_amp`, zonal mean plus diurnal shape |
| District-heating emission factor | 0.03-0.20 kg/kWh | 2023 | Zonal district-heating intensities | `ZoneMarket.ef_dh` |
| PV specific yield | 950-1550 kWh/kWp/yr | TMY | JRC PVGIS climate ranges | `config.PV_YIELD`, sizes the PV portfolio to the positive-energy target |
| Heat-pump COP curve | 3.2 at 7 C, +0.10 per C, bounds [1.8, 5.5] | - | EN 14511 datasheet range | `config.COP0/KAPPA_COP/COP_MIN/COP_MAX` |
| Battery round-trip / SoC / C-rate | 0.90 / [0.15, 0.85] / 0.25 | - | Li-ion datasheet typical | `config.ETA_CH/ETA_DIS/SOC_MIN/SOC_MAX/BATT_C_RATE` |
| Battery degradation cost | 0.02 EUR/kWh cycled | - | Levelized cycle-life estimate | `config.DEG_COST` |
| Thermal storage eff. / loss / C-rate | 0.95 / 0.01 per h / 0.5 | - | Water-tank thermal-storage typical | `config.ETA_TES/KAPPA_TES/TES_C_RATE` |
| Shiftable load share / depth / window | 0.25 / 0.30 / 3 h | - | Demand-response literature range | load split in `data.py`, `config.ALPHA_SHIFT/W_SHIFT` |
| Forecast error s.d. (private / shared) | 0.25 / 0.08 | - | Day-ahead PV and load forecast skill | `model.ERR_PRIV/ERR_SHARED` |
| Hosting limit (net export) | 130-260 kW | - | About 1/3 to 2/3 of a 400 kVA secondary transformer | `Pilot.hosting_limit_kw` |
| Installed capacities, floor areas | per configuration | - | Sized to the positive-energy threshold | `config` agents, `data.build_pilot_data` |

## National remuneration schemes referenced for the export ratios

These are named for traceability. The export ratios are stylized values inspired
by the schemes in force in 2023-24, not exact reproductions.

| Zone | Scheme | Note |
|---|---|---|
| Denmark (DK2) | Hourly spot-based settlement (Danish Energy Agency) | |
| Sweden (SE3) | Microproduction tax reduction (Skatteverket) | In effect through 2025; removed from 1 January 2026 |
| Ireland (SEM) | Clean Export Guarantee (CRU) | |
| Germany | EEG feed-in tariff (Bundesnetzagentur) | |
| Austria | Market-premium scheme (OeMAG / EAG) | |
| Turkey | Unlicensed-generation settlement (EPDK) | |
| Romania | Prosumer regulation (ANRE) | |

## P2P price spread

The modeled peer-to-peer buy-sell spread of roughly 0.04-0.06 EUR/kWh is within
the range reported for community-trading pilots. It is a calibration reference,
not an exact reproduction of any single demonstrator.

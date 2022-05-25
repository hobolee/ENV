# ADMS Details

## General

- The path to ADMS files is /home/dataop/data/nmodel/adms
- Version hkv20b is for forecasting, while hkv23b is for backcasting
- Description of each version can be found on the [EVNF database](http://envf.ust.hk/dataview/adms/current/data_description.py)
- Files are in .nc format. In Python, use the `xarray` or `netcdf4`
  package

## hkv20b

- Use the files under `fcstout`/`fcstout.archive`, they are after bias
  correction
- Files under `data` are before bias-correction (referred as version hkv20)
- `data` directory is linked to `hkv20`

### hkv20b and hkv20 differences

### Points

- Aside from before / after bias-correction
- hkv20 has 18 more points
- Called **receptors**, corresponds to air-quality stations
- They are removed in hkv23b

### Variables

- Share different variables
- hkv20 has point locations, point names, point type, time, etc
- Air quality variables are called Dataset1 - 6
- hkv23b only contains air quality variables, named by its actual name
- hkv23b also includes AQHI

### Point locations

- An _lcc_ projection is used
- The coordinates in WGS84 (Latitude and Longitude) can be found in `config`
- Coordinates for hkv20 (more point): `config/lnglat.npz`
- Coordinates for hkv20b (less point): `config/lnglat-no-receptors.npz`
- The script for converting _amds_ coordinates and WGS84 is `lccxy2lnglat.py`

### hkv20 Point types

The following table shows the index (before dropping receptors) and the their
type.

<!-- markdownlint-disable MD013 -->

| Point           | Description      | Details         |
| --------------- | ---------------- | --------------- |
| 0 - 73199       | grid             | base grid       |
| 73218 - 1203712 | intelligent grid | road            |
| 73200           | receptor         | MB_A (removed)  |
| 73201           | receptor         | TK_A (removed)  |
| 73202           | receptor         | KT_A (removed)  |
| 73203           | receptor         | EN_A (removed)  |
| 73204           | receptor         | ST_A (removed)  |
| 73205           | receptor         | CB_R (removed)  |
| 73206           | receptor         | CB_Rp (removed) |
| 73207           | receptor         | TP_A (removed)  |
| 73208           | receptor         | MKaR (removed)  |
| 73209           | receptor         | MKaRp (removed) |
| 73210           | receptor         | SP_A (removed)  |
| 73211           | receptor         | CL_R (removed)  |
| 73212           | receptor         | CW_A (removed)  |
| 73213           | receptor         | KC_A (removed)  |
| 73214           | receptor         | TW_A (removed)  |
| 73215           | receptor         | YL_A (removed)  |
| 73216           | receptor         | TM_A (removed)  |
| 73217           | receptor         | TC_A (removed)  |

### Station points

The following table shows the index (after removing receptors) which is used by
the ENVF database for making forecast at stations.

| Station | Point   |
| ------- | ------- |
| CW_A    | 1033932 |
| EN_A    | 301658  |
| KC_A    | 598466  |
| KT_A    | 74565   |
| SP_A    | 254774  |
| TW_A    | 820736  |
| ST_A    | 1075441 |
| TP_A    | 658319  |
| TC_A    | 1128282 |
| TM_A    | 856028  |
| YL_A    | 51036   |
| TK_A    | 127816  |
| MB_A    | 55781   |
| CB_R    | 1036915 |
| CL_R    | 341275  |
| MKaR    | 221773  |


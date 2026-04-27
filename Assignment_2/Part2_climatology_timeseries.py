import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4 as nc
from datetime import datetime, timedelta
import os

NC_FILE   = "CRU_Africa_ts4.05.1901.2020.pre.nc"
OUT_DIR   = "part2"
os.makedirs(OUT_DIR, exist_ok=True)

LON_MIN, LON_MAX = 19, 30
LAT_MIN, LAT_MAX =  -28,  -17

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3,  4, 5],
    "JJA": [6,  7, 8],
    "SON": [9, 10, 11],
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

SEASON_COLORS = {
    "DJF": "#4292C6",
    "MAM": "#41AB5D",
    "JJA": "#FE9929",
    "SON": "#CB181D",
}

ds   = nc.Dataset(NC_FILE)
lon  = ds.variables["lon"][:]
lat  = ds.variables["lat"][:]
time = ds.variables["time"]        

origin   = datetime(1900, 1, 1)
dates    = [origin + timedelta(days=float(t)) for t in time[:]]
months   = np.array([d.month for d in dates])
years    = np.array([d.year  for d in dates])

precip   = ds.variables["pre"][:]  
ds.close()
print(f"  Dimensions – time:{len(dates)}  lat:{len(lat)}  lon:{len(lon)}")

lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]

precip_sn = precip[:, lat_idx[:, None], lon_idx]   # (time, lat_sn, lon_sn)

precip_sn_mean = np.ma.mean(precip_sn.reshape(len(dates), -1), axis=1)
print(f"  Botswana pixels: {len(lat_idx) * len(lon_idx)}")


# MEAN MONTHLY CLIMATOLOGY
mean_clim = np.array([
    np.ma.mean(precip_sn_mean[months == m]) for m in range(1, 13)])

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(MONTH_LABELS, mean_clim,
              color=plt.cm.Blues(np.linspace(0.35, 0.85, 12)),
              edgecolor="white", linewidth=0.6)
ax.set_title("Mean Monthly Climatology – Botswana (1901–2020)\n"
             "CRU TS4.05 Precipitation", fontsize=13, fontweight="bold")
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Precipitation (mm/month)", fontsize=11)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)

# Value labels on bars
for bar, val in zip(bars, mean_clim):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
out = os.path.join(OUT_DIR, "mean_climatology_botswana.png")
plt.savefig(out, dpi=150)
plt.close()


# SEASONAL CLIMATOLOGY  (DJF, MAM, JJA, SON)
seasonal_means = {
    season: np.ma.mean(precip_sn_mean[np.isin(months, mlist)])
    for season, mlist in SEASONS.items()}

fig, ax = plt.subplots(figsize=(7, 5))
for i, (season, val) in enumerate(seasonal_means.items()):
    ax.bar(season, val,
           color=SEASON_COLORS[season],
           edgecolor="white", linewidth=0.6,
           label=season, width=0.55)
    ax.text(i, val + 1, f"{val:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("Seasonal Climatology – Botswana (1901–2020)\n"
             "CRU TS4.05 Precipitation", fontsize=13, fontweight="bold")
ax.set_xlabel("Season", fontsize=11)
ax.set_ylabel("Mean Precipitation (mm/month)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()
out = os.path.join(OUT_DIR, "seasonal_climatology_botswana.png")
plt.savefig(out, dpi=150)
plt.close()


# ANNUAL PRECIPITATION TIME SERIES
unique_years = np.unique(years)
annual_total = np.array([
    np.ma.sum(precip_sn_mean[years == y]) for y in unique_years])

# Linear trend
trend_coef = np.polyfit(unique_years, annual_total, 1)
trend_line = np.polyval(trend_coef, unique_years)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(unique_years, annual_total,
        color="#2171B5", linewidth=1.2, label="Annual total")
ax.fill_between(unique_years, annual_total,
                alpha=0.15, color="#2171B5")
ax.plot(unique_years, trend_line,
        color="firebrick", linewidth=1.5, linestyle="--",
        label=f"Linear trend ({trend_coef[0]:+.2f} mm/yr)")

ax.set_title("Annual Precipitation Time Series – Botswana (1901–2020)\n"
             "CRU TS4.05 Precipitation", fontsize=13, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Total Annual Precipitation (mm/year)", fontsize=11)
ax.legend(frameon=False, fontsize=10)
ax.grid(linestyle="--", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(OUT_DIR, "annual_timeseries_botswana.png")
plt.savefig(out, dpi=150)
plt.close()

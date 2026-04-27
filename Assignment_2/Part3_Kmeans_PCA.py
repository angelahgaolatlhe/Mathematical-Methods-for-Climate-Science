import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm

# =============================================================================
# LOAD DATA
# =============================================================================

file_path = 'CRU_Africa_ts4.05.1901.2020.pre.nc'
ds        = xr.open_dataset(file_path)

precip = ds['pre']
lat    = ds['lat']
lon    = ds['lon']

# Handle missing values
precip = precip.where(precip != 9.96921e+36, np.nan)

# Select full period
precip = precip.sel(time=slice('1901-01-01', '2020-12-31'))

n_time, n_lat, n_lon = len(precip.time), len(precip.lat), len(precip.lon)

# Reshape: (pixels × time)
precip_reshaped = precip.values.reshape(n_time, n_lat * n_lon).T
print("Data shape (pixels × time):", precip_reshaped.shape)

# Land mask
land_mask = np.isfinite(precip.mean(dim='time').values).ravel()
X_land    = precip_reshaped[land_mask, :]
land_idx  = np.where(land_mask)[0]

# Standardize
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(np.nan_to_num(X_land))
print("Standardized shape:", X_scaled.shape)

# Meshgrid for plotting
lon2d, lat2d = np.meshgrid(lon, lat)

# =============================================================================
# SECTION 1 – PCA
# =============================================================================

N_COMPONENTS = 10
pca          = PCA(n_components=N_COMPONENTS)
X_pca        = pca.fit_transform(X_scaled)       # (n_land, N_COMPONENTS) – spatial scores
explained_variance = pca.explained_variance_ratio_
print("Explained variance:", explained_variance)

# Transpose PCA: (time × pixels) → time-series scores
pca_t   = PCA(n_components=N_COMPONENTS)
X_pca_t = pca_t.fit_transform(X_scaled.T)        # (n_time, N_COMPONENTS)

# ── Plot 1: Scree plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(1, N_COMPONENTS + 1),
            explained_variance * 100,
            color='#2171B5', edgecolor='white')
axes[0].set_title("Variance Explained per PC", fontweight='bold')
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0].spines[['top', 'right']].set_visible(False)
for i, v in enumerate(explained_variance * 100):
    axes[0].text(i + 1, v + 0.3, f"{v:.1f}%", ha='center', fontsize=10)

axes[1].plot(range(1, N_COMPONENTS + 1),
             np.cumsum(explained_variance) * 100,
             marker='o', color='#2171B5', linewidth=2)
axes[1].set_title("Cumulative Variance Explained", fontweight='bold')
axes[1].set_xlabel("Number of PCs")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend(frameon=False)
axes[1].grid(linestyle='--', alpha=0.4)
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle("PCA Scree Plot – CRU Precipitation (1901–2020)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Plot 2: Spatial maps of PC1 and PC2 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                          subplot_kw={'projection': ccrs.PlateCarree()})

for i, ax in enumerate(axes):
    pc_spatial        = np.full(n_lat * n_lon, np.nan)
    pc_spatial[land_idx] = X_pca[:, i]
    pc_pattern        = pc_spatial.reshape(n_lat, n_lon)

    ax.set_facecolor('lightblue')
    vmax = np.nanpercentile(np.abs(pc_pattern), 98)
    im   = ax.pcolormesh(lon2d, lat2d, pc_pattern,
                         cmap='coolwarm', vmin=-vmax, vmax=vmax,
                         transform=ccrs.PlateCarree(), shading='auto')
    plt.colorbar(im, ax=ax, pad=0.03, shrink=0.85, label=f'PC{i+1} Score')
    ax.coastlines(linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.set_xticks(np.arange(-20, 54, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-40, 41, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
    ax.text(0.02, 0.97,
            f"Var. explained: {explained_variance[i]*100:.1f}%",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.set_title(f"Spatial Pattern of PC{i+1}\n(CRU Precipitation, 1901–2020)",
                 fontsize=12, fontweight='bold')

    # Save pc1_pattern for later use in comparison plot
    if i == 0:
        pc1_pattern = pc_pattern

plt.tight_layout()
plt.savefig("pca_spatial_pc1_pc2.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Plot 3: PC1 and PC2 time series ──────────────────────────────────────────
time_index = np.arange(n_time)

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(time_index, X_pca_t[:, i],
            color='#2171B5' if i == 0 else '#E6550D', linewidth=0.9)
    ax.set_title(f"PC{i+1} Time Series", fontweight='bold')
    ax.set_ylabel(f"PC{i+1} Score")
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.01, 0.95,
            f"Var. explained: {explained_variance[i]*100:.1f}%",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

axes[-1].set_xlabel("Time Index (months from 1901-01)")
plt.suptitle("PC1 & PC2 Time Series – CRU Precipitation (1901–2020)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("pca_timeseries.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# SECTION 2 – K-Means Clustering
# =============================================================================

# ── Plot 4: Elbow + Silhouette ────────────────────────────────────────────────
sample_size = min(5000, X_scaled.shape[0])
rng         = np.random.default_rng(42)
sample_idx  = rng.choice(X_scaled.shape[0], sample_size, replace=False)
X_sample    = X_scaled[sample_idx]

k_range    = range(2, 13)
inertias   = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_sample)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sample, km.labels_))
    print(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={sil_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker='o', color='#2171B5', linewidth=2)
axes[0].set_title("Elbow Method – Inertia", fontweight='bold')
axes[0].set_xlabel("Number of Clusters k")
axes[0].set_ylabel("Inertia (Within-cluster SS)")
axes[0].grid(linestyle='--', alpha=0.4)
axes[0].spines[['top', 'right']].set_visible(False)

axes[1].plot(list(k_range), sil_scores, marker='s', color='#41AB5D', linewidth=2)
axes[1].set_title("Silhouette Score", fontweight='bold')
axes[1].set_xlabel("Number of Clusters k")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(linestyle='--', alpha=0.4)
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle("K-Means Cluster Selection – CRU Precipitation (1901–2020)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Plot 5: Spatial cluster map ───────────────────────────────────────────────
n_clusters     = 3
kmeans         = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)

cluster_map           = np.full(n_lat * n_lon, np.nan)
cluster_map[land_idx] = cluster_labels
masked_cluster_map    = cluster_map.reshape(n_lat, n_lon)

colors = ['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b',
          '#a6d96a','#66bd63','#1a9850','#006837','#08306b']
physically_meaningful_cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1),
                   physically_meaningful_cmap.N)

plt.figure(figsize=(14, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')

cluster_plot = ax.pcolormesh(lon2d, lat2d, masked_cluster_map,
                              cmap=physically_meaningful_cmap, norm=norm,
                              transform=ccrs.PlateCarree(), shading='auto')

cbar = plt.colorbar(cluster_plot, ax=ax, pad=0.05)
cbar.set_ticks(np.arange(0, n_clusters, 1))
cbar.set_ticklabels([f'Cluster {i}' for i in range(n_clusters)])
cbar.set_label('K-Means Cluster Index', fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.coastlines(linewidth=1.5)
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='gray')
ax.add_feature(cfeature.LAND,    edgecolor='black')
ax.add_feature(cfeature.OCEAN,   facecolor='lightblue')
ax.set_xticks(np.arange(-30, 60, 10), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-40, 40, 10), crs=ccrs.PlateCarree())
ax.tick_params(labelsize=12, labelcolor='black')
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))

plt.title("K-Means Clustering Map of Precipitation Patterns (1901–2020)",
          fontsize=16, fontweight='bold')
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude',  fontsize=14)
plt.tight_layout()
plt.savefig("kmeans_map.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# SECTION 3 – Comparison: PC1 spatial pattern vs K-Means cluster map
# =============================================================================

# ── Plot 6: Side-by-side comparison ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                          subplot_kw={'projection': ccrs.PlateCarree()})

# Left: PC1 spatial pattern
ax1 = axes[0]
ax1.set_facecolor('lightblue')
vmax = np.nanpercentile(np.abs(pc1_pattern), 98)
im1  = ax1.pcolormesh(lon2d, lat2d, pc1_pattern,
                       cmap='coolwarm', vmin=-vmax, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading='auto')
plt.colorbar(im1, ax=ax1, pad=0.03, shrink=0.85, label='PC1 Score')
ax1.coastlines(linewidth=1.2)
ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='gray')
ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax1.set_xticks(np.arange(-20, 54, 10), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-40, 41, 10), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
ax1.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
ax1.set_title("PCA – PC1 Spatial Pattern\n(CRU Precipitation, 1901–2020)",
              fontsize=12, fontweight='bold')
ax1.text(0.02, 0.97,
         f"Var. explained: {explained_variance[0]*100:.1f}%",
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Right: K-Means cluster map
ax2 = axes[1]
ax2.set_facecolor('lightblue')
im2 = ax2.pcolormesh(lon2d, lat2d, masked_cluster_map,
                      cmap=physically_meaningful_cmap, norm=norm,
                      transform=ccrs.PlateCarree(), shading='auto')
cbar2 = plt.colorbar(im2, ax=ax2, pad=0.03, shrink=0.85)
cbar2.set_ticks(np.arange(0, n_clusters, 1))
cbar2.set_ticklabels([f'C{i}' for i in range(n_clusters)], fontsize=9)
cbar2.set_label('Cluster Index', fontsize=11)
ax2.coastlines(linewidth=1.2)
ax2.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='gray')
ax2.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax2.set_xticks(np.arange(-20, 54, 10), crs=ccrs.PlateCarree())
ax2.set_yticks(np.arange(-40, 41, 10), crs=ccrs.PlateCarree())
ax2.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
ax2.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
ax2.set_title(f"K-Means Clustering (k={n_clusters})\n(CRU Precipitation, 1901–2020)",
              fontsize=12, fontweight='bold')

plt.suptitle("Comparison: PCA vs K-Means – CRU African Precipitation (1901–2020)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("comparison_pca_kmeans.png", dpi=300, bbox_inches="tight")
plt.show()

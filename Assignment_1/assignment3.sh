#!/bin/bash

INPUT="CRU_Africa_ts4.05.1901.2020.pre.nc"

# Select time period
cdo selyear,1991/2020 $INPUT period.nc

# Select JJA season
cdo selmon,6,7,8 period.nc jja.nc

# Compute climatology
cdo timmean jja.nc jja_clim.nc

# Compute anomalies
cdo sub jja.nc jja_clim.nc jja_anomaly.nc

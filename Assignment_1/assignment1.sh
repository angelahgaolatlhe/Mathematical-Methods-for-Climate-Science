#!/bin/bash
URL="https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/"
SAVE_DIR="PRECIP"

mkdir -p "$SAVE_DIR"

for year in {1983..2016}
do
    wget -P "$SAVE_DIR" "${URL}chirps-v2.0.${year}.days_p25.nc"
done



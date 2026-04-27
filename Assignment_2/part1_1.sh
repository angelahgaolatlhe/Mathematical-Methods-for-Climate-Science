#!/bin/bash
mkdir -p cpc_precip
cd cpc_precip

URL="https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/"

for year in {1979..1988}; 
do
    wget "${URL}precip.${year}.nc"
done

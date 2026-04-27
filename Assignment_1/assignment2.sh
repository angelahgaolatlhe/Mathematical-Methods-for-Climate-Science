#!/bin/bash
INPUT="GMFD_Africa_tas_daily_1948_2008.nc"
TMP="tas_celsius.nc"
OUTPUT="hot_days.nc"

# Convert Kelvin to Celsius
cdo subc,273.15 $INPUT $TMP

# Get temperatures greater than 35 degrees
cdo gec,35 $TMP mask.nc    

# Count the number of unique days  
cdo timsum mask.nc $OUTPUT

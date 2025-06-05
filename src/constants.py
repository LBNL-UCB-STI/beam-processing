TMP_DIR = ".tmp"
CONGESTION_THRESHOLD_MPH = 2.0
FUEL_CONVERSION_JOULES_GALLON_GASOLINE = 8.3141841e-9  # Should be MJ/gallon * J/MJ # This was incorrect, 1 MJ = 1e6 J, 1 Gallon Gasoline ~ 120 MJ. 1 Joule = 1e-6 MJ. (1 gal * 120 MJ/gal * 1e6 J/MJ)^-1 ~ 8.33e-9 gal/Joule. The factor seems correct for JOULES TO GALLONS conversion.
FUEL_CONVERSION_JOULES_GALLON_DIESEL = (
    8.3141841e-9  # MJ/gallon * J/MJ # Same note as above.
)
FUEL_CONVERSION_JOULES_KWH = 3.6e6  # J/kWh
ELECTRICITY_EMISSION_KG_PER_KWH = (
    0.0005  # kg CO2 / kWh (Example value, specific to region/grid)
)
OMX_FILE_NAME = "skims.omx"

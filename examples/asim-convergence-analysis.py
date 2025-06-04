from src import input, outputDataDirectory
import pandas as pd
import geopandas as gpd
import os

# import matplotlib
#
# matplotlib.use("TkAgg")


os.chdir("../")
# "https://storage.googleapis.com/beam-core-outputs/seattle-util-diff-20240715/activitysim/final_trips.csv"


# outputPath = "https://storage.googleapis.com/beam-core-outputs/seattle-util-diff-20240715"
#
# pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
#     outputPath,
#     years=[2010,2012, 2014, 2016, 2018, 2020],
#     asimLiteIterations=2,
#     beamIterations=0,
#     region="Seattle",
# )

outputPath = (
    "https://storage.googleapis.com/beam-core-outputs/seattle-util-diff-20240715"
)

pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
    outputPath,
    years=[2017, 2018, 2019],
    asimLiteIterations=2,
    beamIterations=0,
    region="Seattle",
)

pilatesData_new = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/seattle-2010-base"),
    pilatesDirectory,
    region="Seattle",
)

# look = pilatesData_new.beamRuns[(2018,2)].realizedModeCount.dataFrame
pilatesData_new.scoreStatsByIteration.clearCache()
mc_new = pilatesData_new.tripModeCountPerIteration.dataFrame
scores_new = pilatesData_new.scoreStatsByIteration.dataFrame
mc_new = mc_new.unstack()["count"]
pilatesData_new.replanningEventReasonPerIteration.clearCache()
repl_new = pilatesData_new.replanningEventReasonPerIteration.dataFrame
# pmt_new = pilatesData_new.passengerMilesByVehicleAndModeByIteration.dataFrame
# rmc_new = pilatesData_new.realizedModeCountyByIteration.dataFrame.unstack()['mode']
look = pilatesData_new.congestionInfoByYear.dataFrame

outputPath = "gs://beam-core-outputs/seattle-no-plans-20240716"

pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
    outputPath,
    years=[2010, 2012, 2014, 2016],
    asimLiteIterations=2,
    beamIterations=0,
    region="Seattle",
)

pilatesData_noplans = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/seattle-2010-noplans"),
    pilatesDirectory,
    region="Seattle",
)
pilatesData_noplans.tripModeCountPerIteration.clearCache()
pilatesData_noplans.scoreStatsByIteration.clearCache()
mc_noplans = pilatesData_noplans.tripModeCountPerIteration.dataFrame
scores_noplans = pilatesData_noplans.scoreStatsByIteration.dataFrame
mc_noplans = mc_noplans.unstack()["count"]
repl_noplans = pilatesData_noplans.replanningEventReasonPerIteration.dataFrame
# pmt_noplans = pilatesData_noplans.passengerMilesByVehicleAndModeByIteration.dataFrame
# rmc_noplans = pilatesData_noplans.realizedModeCountyByIteration.dataFrame.unstack()['mode']
print("Stop")

walktransit_new = pmt_new["walk_transit"].unstack()
walktransit_noplans = pmt_noplans["walk_transit"].unstack()

car_new = pmt_new["car"].unstack()
car_noplans = pmt_noplans["car"].unstack()
drive_transit_new = pmt_new["drive_transit"].unstack()
drive_transit_noplans = pmt_noplans["drive_transit"].unstack()

stacked_rmc = rmc_noplans.stack()
stacked_rmc.index.set_names(["year", "iteration", "currentTourMode"], inplace=True)
look = (
    pmt_noplans.stack()
    .unstack("vehicleType")
    .divide(stacked_rmc, axis=0)
    .unstack("currentTourMode")
)
outputPath = "gs://beam-core-outputs/seattle-newplans-20240606"

pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
    outputPath,
    years=[2010],
    asimLiteIterations=2,
    beamIterations=0,
    region="Seattle",
)

pilatesData_newplans = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/seattle-2010-newpllans"),
    pilatesDirectory,
    region="Seattle",
)
mc_newplans = pilatesData_newplans.tripModeCountPerIteration.dataFrame
scores_newplans = pilatesData_newplans.scoreStatsByIteration.dataFrame
vmt_newplans = pilatesData_newplans.modeVMTPerYear.dataFrame
print("Stop")

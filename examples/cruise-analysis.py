from src import input, outputDataDirectory
import pandas as pd
import geopandas as gpd
import os

# import matplotlib
#
# matplotlib.use("TkAgg")

os.chdir("../")

outputPath = "gs://cruise-outputs/sfbay_cruise_SAVBaseline_1_phase2_66"

pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
    outputPath,
    years=[2018,2019,2020],
    asimLiteIterations=3,
    beamIterations=1,
    region="SFBay",
)

pilatesData = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/cruise-base"),
    pilatesDirectory,
    region="SFBay",
)
tmc = pilatesData.tourModeCountPerIteration.dataFrame['count'].unstack()
mc = pilatesData.tripModeCountPerIteration.dataFrame['count'].unstack()

outputPath2 = "https://storage.cloud.google.com/cruise-outputs/sfbay_cruise_SAVBaseline_1_phase2_65"

pilatesDirectory2 = outputDataDirectory.PilatesRunInputDirectory(
    outputPath2,
    years=[2018, 2019, 2020],
    asimLiteIterations=3,
    beamIterations=1,
    region="SFBay",
)

pilatesData2 = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/cruise-rhtransit"),
    pilatesDirectory2,
    region="SFBay",
)
ss = pilatesData2.scoreStatsByIteration.dataFrame

tmc2 = pilatesData2.tourModeCountPerIteration.dataFrame['count'].unstack()
mc2 = pilatesData2.tripModeCountPerIteration.dataFrame['count'].unstack()
mc = pilatesData.tripModeCountPerIteration.dataFrame
look2 = pilatesData.scoreStatsByIteration.dataFrame

tmc = pilatesData.asimRuns[(2011, 2)].tripModeCountByOrigin.toGdf()
print("Stop")

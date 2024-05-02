from src import input, outputDataDirectory
import pandas as pd
import geopandas as gpd
import os

# import matplotlib
#
# matplotlib.use("TkAgg")

os.chdir("../")

outputPath = "gs://beam-core-outputs/seattle-fillskims-20240429"

pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
    outputPath,
    years=[2010,2012],
    asimLiteIterations=2,
    beamIterations=0,
    region="Seattle",
)

pilatesData_new = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/seattle-2010-base"),
    pilatesDirectory,
    region="Seattle",
)
mc_new = pilatesData_new.tripModeCountPerIteration.dataFrame
scores_new = pilatesData_new.scoreStatsByIteration.dataFrame

outputPath = "gs://beam-core-outputs/seattle-fillskims-20240427"

pilatesDirectory = outputDataDirectory.PilatesRunInputDirectory(
    outputPath,
    years=[2010,2012],
    asimLiteIterations=2,
    beamIterations=0,
    region="Seattle",
)

pilatesData_old = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory("output/seattle-2010-old"),
    pilatesDirectory,
    region="Seattle",
)
mc_old = pilatesData_old.tripModeCountPerIteration.dataFrame
scores_old = pilatesData_old.scoreStatsByIteration.dataFrame
print("Stop")

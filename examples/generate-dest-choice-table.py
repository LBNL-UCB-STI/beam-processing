from src import input, outputDataDirectory
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

from src.input import PilatesRunInputDirectory
from src.outputDataDirectory import PilatesOutputData, OutputDataDirectory

os.chdir("../")

scenarioName = "base"
years = [2010]
asimLiteIterations = 2
beamIterations = 2
folderName = (
    "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-{0}-20231211".format(
        scenarioName
    )
)


directory = PilatesRunInputDirectory(
    folderName, [2010], 2, 2
)
run = PilatesOutputData(
    OutputDataDirectory("output/{0}".format("base-gen")), directory
)

tours = run.asimRuns[(2010,2)].tours.dataFrame
hh = run.asimRuns[(2010,2)].households.dataFrame
tours = tours[['person_id','tour_type','origin','destination','household_id','tour_mode']].copy()
tours.tour_mode = tours.tour_mode.replace({"SHARED3FREE":"DRIVE","SHARED3PAY":"DRIVE","SHARED2FREE":"DRIVE","SHARED2PAY":"DRIVE","DRIVEALONEFREE":"DRIVE","DRIVEALONEPAY":"DRIVE"})
tours = tours.loc[tours.tour_mode.isin(["DRIVE","WALK","WALK_LOC","WALK_HVY","WALK_COM","WALK_LRF"]), :]
t = tours.head(1000)
skims = run.skims.dataFrame
skim_values = skims.unstack().reindex(t.origin.astype(int))
chosen_outcomes = skims.reindex(pd.MultiIndex.from_frame(t[['origin','destination']].astype(int)))
for c in chosen_outcomes.columns:
    t[c] = chosen_outcomes[c].values
print("done")
